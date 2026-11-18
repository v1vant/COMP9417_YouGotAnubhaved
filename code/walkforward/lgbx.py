"""Pure test (not in the repo): LightGBM price models beyond the plain up/down classifier.

Same hyperparameters as train.fit_model's price LightGBM (min_child_samples 160, colsample/subsample 0.85,
reg_lambda 8, max_bin 127, seed 9417, deterministic) and the same class balancing ("balanced"), so a
'classify' spec with no extras reproduces the package classifier.

task 'regress' (continuous price margin m = price(D+1,t) - mean(D)):
  target 'slog' = sign(m) log(1 + |m|);  'rel' = clip(m / max(|thr|, 10), -3, 3)
  objective 'l2' | 'l1' | 'huber' (delta 1 on the transformed scale); rows class-balanced by sign(m),
  so the natural decision rule "predicted margin > 0" plays the role of probability > 0.5.
  Reported probability = sigmoid(prediction / sd(training target)) (only the sign matters at 0.5).
task 'classify':
  loss 'logloss' | 'focal' (gamma); optional extra row weights (class totals kept balanced):
  'oof_uncertainty': 1 + alpha * 4 p (1 - p), p = out-of-sample LightGBM probability (hard for past models)
  'near_threshold' : 1 + alpha * exp(-|m| / (0.05 * max(|thr|, 10)))   (price ends close to the threshold)
  'abs_margin'     : 1 + alpha * min(|m| / max(|thr|, 10), 3)           (contrast: far rows count more)
"""
import numpy as np
import lightgbm as lgb

SEED = 9417
BASE = dict(min_child_samples=160, colsample_bytree=0.85, subsample=0.85, subsample_freq=1, reg_lambda=8.0,
            max_bin=127, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True, random_state=SEED)


def balanced(y):
    y = np.asarray(y, bool)
    w = np.where(y, len(y) / (2 * y.sum()), len(y) / (2 * (~y).sum()))
    return w


def rebalance(w, y):
    """Scale weights within each class so both classes keep equal total weight (as 'balanced')."""
    y = np.asarray(y, bool); w = np.asarray(w, np.float64).copy()
    w[y] *= (len(y) / 2) / w[y].sum(); w[~y] *= (len(y) / 2) / w[~y].sum()
    return w


def focal_objective(gamma, surrogate=False):
    def grad_of(z, y):
        p = 1 / (1 + np.exp(-z))
        lp, lq = np.log(np.clip(p, 1e-12, 1)), np.log(np.clip(1 - p, 1e-12, 1))
        g1 = gamma * p * (1 - p) ** gamma * lp - (1 - p) ** (gamma + 1)
        g0 = -gamma * p ** gamma * (1 - p) * lq + p ** (gamma + 1)
        return np.where(y > 0.5, g1, g0)

    def objective(preds, data):
        y, w = data.get_label(), data.get_weight()
        eps = 1e-4
        g = grad_of(preds, y)
        h = (grad_of(preds + eps, y) - grad_of(preds - eps, y)) / (2 * eps)
        if surrogate:     # true curvature is negative for confidently wrong rows; use a positive stand-in there
            pz = 1 / (1 + np.exp(-preds)); pt = np.where(y > 0.5, pz, 1 - pz)
            h = np.maximum(h, (1 - pt) ** gamma * pz * (1 - pz))
        h = np.maximum(h, 1e-6)
        return g * w, h * w
    return objective


def fit_predict(spec, Xtr, Xpr, label_tr, margin_tr, thr_tr, oof_p_tr=None, target_tr=None):
    c = spec["config"]
    global BASE
    base_saved = BASE
    BASE = dict(BASE, **spec.get("lgb_params", {}), **({"random_state": spec["seed"]} if "seed" in spec else {}))
    try:
        return _fit_predict(spec, c, Xtr, Xpr, label_tr, margin_tr, thr_tr, oof_p_tr, target_tr)
    finally:
        BASE = base_saved


def _fit_predict(spec, c, Xtr, Xpr, label_tr, margin_tr, thr_tr, oof_p_tr=None, target_tr=None):
    y = np.asarray(label_tr, bool)
    task = spec.get("task", "classify")
    if task == "regress":
        m = np.asarray(margin_tr, np.float64)
        if spec["target"] == "slog":
            t = np.sign(m) * np.log1p(np.abs(m))
        elif spec["target"] == "given":          # e.g. within-day percentile rank (layered design)
            t = np.asarray(target_tr, np.float64)
        elif spec["target"] == "rel":
            t = np.clip(m / np.maximum(np.abs(thr_tr), 10.0), -3, 3)
        else:
            raise ValueError(spec["target"])
        obj = {"l2": "regression", "l1": "regression_l1", "huber": "huber"}[spec["objective"]]
        est = lgb.LGBMRegressor(objective=obj, n_estimators=c["trees"], learning_rate=c["learning_rate"],
                                num_leaves=c["leaves"], **({"alpha": 1.0} if obj == "huber" else {}), **BASE)
        est.fit(Xtr, t, sample_weight=balanced(y))
        raw = np.concatenate([est.predict(Xpr.iloc[i:i + 20000]) for i in range(0, len(Xpr), 20000)])
        scale = float(np.std(t))
        centre = 0.5 if spec["target"] == "given" else 0.0
        return 1 / (1 + np.exp(-np.clip((raw - centre) / scale, -60, 60))), raw
    w = balanced(y)
    extra = spec.get("weights")
    if extra:
        a = spec.get("alpha", 2.0)
        m, thr = np.asarray(margin_tr, np.float64), np.asarray(thr_tr, np.float64)
        if extra == "oof_uncertainty":
            p = np.asarray(oof_p_tr, np.float64); h = 1 + a * 4 * p * (1 - p)
        elif extra == "near_threshold":
            h = 1 + a * np.exp(-np.abs(m) / (0.05 * np.maximum(np.abs(thr), 10.0)))
        elif extra == "abs_margin":
            h = 1 + a * np.minimum(np.abs(m) / np.maximum(np.abs(thr), 10.0), 3.0)
        else:
            raise ValueError(extra)
        w = rebalance(w * h, y)
    if spec.get("loss", "logloss") == "focal":
        params = dict(objective=focal_objective(spec["gamma"], spec.get("hess") == "surrogate"), learning_rate=c["learning_rate"],
                      num_leaves=c["leaves"], min_data_in_leaf=160, feature_fraction=0.85, bagging_fraction=0.85,
                      bagging_freq=1, lambda_l2=8.0, max_bin=127, num_threads=2, verbosity=-1, deterministic=True,
                      force_col_wise=True, seed=SEED)
        booster = lgb.train(params, lgb.Dataset(Xtr, label=y.astype(float), weight=w), num_boost_round=c["trees"])
        raw = np.concatenate([booster.predict(Xpr.iloc[i:i + 20000], raw_score=True) for i in range(0, len(Xpr), 20000)])
        return 1 / (1 + np.exp(-np.clip(raw, -60, 60))), raw
    est = lgb.LGBMClassifier(objective="binary", n_estimators=c["trees"], learning_rate=c["learning_rate"],
                             num_leaves=c["leaves"], **BASE)
    est.fit(Xtr, y.astype(int), sample_weight=w)
    prob = np.concatenate([est.predict_proba(Xpr.iloc[i:i + 20000])[:, 1] for i in range(0, len(Xpr), 20000)])
    return prob, None


def rank_fit_predict(spec, Xtr, Xpr, labels_tr, groups_tr):
    """LightGBM LambdaRank over the slots of each day (labels = within-day price decile 0..9)."""
    c = spec["config"]
    params = {k: v for k, v in BASE.items()}
    est = lgb.LGBMRanker(objective="lambdarank", n_estimators=c["trees"], learning_rate=c["learning_rate"],
                         num_leaves=c["leaves"], label_gain=list(range(10)), lambdarank_truncation_level=48, **params)
    est.fit(Xtr, np.asarray(labels_tr, int), group=np.asarray(groups_tr, int))
    raw = np.concatenate([est.predict(Xpr.iloc[i:i + 20000]) for i in range(0, len(Xpr), 20000)])
    return 1 / (1 + np.exp(-np.clip(raw, -60, 60))), raw
