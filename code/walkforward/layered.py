"""Pure test (not in the repo): the layered price-direction design.

Decomposition (target day D+1, threshold = mean price of day D):
  up_t  <=>  price(D+1,t) - mean(D) > 0  <=>  t is among the N(D+1) slots of D+1 above mean(D),
  where N = count of up slots. If slots are ranked by tomorrow's price, the up slots are exactly the top N.

Layer 1 (day model, one row per complete day): predicts N (mean, 10/50/90% quantiles) and the level shift
  L = mean(D+1) - mean(D) (signed log), from day-constant features only: today's/last week's/4-week price and
  demand profiles and stats, interstate stats, tomorrow's weather summaries, calendar, tomorrow's demand-forecast
  day summary and 48-slot profile, and three persistence counts (slots above today's mean today / last week /
  4-week profile).
Layer 2 (intraday model): LightGBM regression of each slot's within-day percentile rank of tomorrow's price.
Layer 3 (combination, in score_layered.py / harness stacking): top-N by LightGBM probability, top-N by the
  intraday score, P(N >= r_t) from the day model's quantiles, and a stacked classifier.
All features are known at forecast time; every target is used only on training days."""
import numpy as np, pandas as pd
import lightgbm as lgb

SEED = 9417
DAY_PARAMS = dict(num_leaves=15, n_estimators=600, learning_rate=0.02, min_child_samples=20, colsample_bytree=0.5,
                  subsample=0.8, subsample_freq=1, reg_lambda=1.0, n_jobs=2, verbosity=-1, deterministic=True,
                  force_col_wise=True, random_state=SEED)


def day_table(X, days):
    """One row per market day (target day), from columns that are constant within the day."""
    d = np.asarray(days)
    complete = pd.Series(1, index=X.index).groupby(d).transform("size").to_numpy() == 48
    first = ~pd.Series(d).duplicated().to_numpy()
    rows = X[first & complete]
    sample = np.asarray(pd.DatetimeIndex(d).year == 2019)          # complete, demand forecasts present
    sd = X[sample].groupby(d[sample]).std().max()
    const = [c for c in X.columns if sd[c] < 1e-6]
    T = rows[const].copy()
    T.index = pd.DatetimeIndex(d[first & complete])
    for name, prefix in (("today", "nsw_price_profile_lag_1_margin_slot_"), ("lastweek", "nsw_price_profile_lag_7_margin_slot_"),
                         ("weekly", "nsw_price_profile_weekly_margin_slot_")):
        cols = [c for c in T.columns if c.startswith(prefix)]
        if len(cols) == 48:
            T[f"count_up_{name}"] = (T[cols] > 0).sum(axis=1)
    return T


def fit_predict_day(T, n_day, level_day, train_days, pred_days, quantiles=(0.1, 0.5, 0.9), level_rel_day=None):
    """Fit the day models on train_days and predict pred_days. Returns a DataFrame indexed by pred day.
    Columns: n (mean count), q<k> for each quantile, l (signed-log level), lrel (relative level, if given)."""
    Xtr, Xpr = T.loc[train_days], T.loc[pred_days]
    out = pd.DataFrame(index=pred_days)
    n = n_day.loc[train_days].to_numpy(np.float64)
    out["n"] = lgb.LGBMRegressor(objective="regression", **DAY_PARAMS).fit(Xtr, n).predict(Xpr)
    for q in quantiles:
        out[f"q{int(round(q*100))}"] = lgb.LGBMRegressor(objective="quantile", alpha=q, **DAY_PARAMS).fit(Xtr, n).predict(Xpr)
    lv = level_day.loc[train_days].to_numpy(np.float64)
    t = np.sign(lv) * np.log1p(np.abs(lv))
    out["l"] = lgb.LGBMRegressor(objective="regression", **DAY_PARAMS).fit(Xtr, t).predict(Xpr)
    if level_rel_day is not None:
        out["lrel"] = lgb.LGBMRegressor(objective="regression", **DAY_PARAMS).fit(
            Xtr, level_rel_day.loc[train_days].to_numpy(np.float64)).predict(Xpr)
    return out


def prob_from_count(score, mu, q10, q90, days):
    """P(up_t) = P(N >= r_t), r_t = rank of slot t by `score` within its day (1 = highest), N ~ Normal(mu, sd)
    with sd from the 10-90% quantile spread (floored at 1 slot)."""
    from scipy.stats import norm
    r = pd.Series(score).groupby(np.asarray(days)).rank(ascending=False, method="first").to_numpy()
    sd = np.maximum((np.asarray(q90) - np.asarray(q10)) / 2.5631, 1.0)
    return 1 - norm.cdf((r - 0.5 - np.asarray(mu)) / sd), r


def top_n(score, n, days):
    """Mark the round(n) highest-scoring slots of each day as up."""
    r = pd.Series(score).groupby(np.asarray(days)).rank(ascending=False, method="first").to_numpy()
    return r <= np.clip(np.round(np.asarray(n)), 0, 48)


def prob_from_quantile_grid(score, Q, qs, days, widen=1.0):
    """P(up_t) = P(N >= r_t) with N's CDF interpolated through the predicted quantiles Q (rows x len(qs)),
    widened around the median by `widen`, padded with F(-0.5)=0 and F(48.5)=1."""
    r = pd.Series(score).groupby(np.asarray(days)).rank(ascending=False, method="first").to_numpy()
    qs = np.asarray(qs, float); Q = np.sort(np.asarray(Q, float), axis=1)
    med = np.array([np.interp(0.5, qs, row) for row in Q])
    Qw = np.clip(med[:, None] + widen * (Q - med[:, None]), -0.5, 48.5)
    out = np.empty(len(r))
    for i in range(len(r)):
        xs = np.concatenate([[-0.5], np.maximum.accumulate(Qw[i]), [48.5]])
        ys = np.concatenate([[0.0], qs, [1.0]])
        out[i] = 1 - np.interp(r[i] - 0.5, xs, ys)
    return out


def count_match(prob, target_count, days, strength=1.0):
    """Shift each day's logits by strength * d, where d makes the day's expected up-count equal target_count."""
    p = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6); z = np.log(p / (1 - p))
    d = np.asarray(days); out = np.empty_like(z)
    tc = np.asarray(target_count, float)
    for day in pd.unique(d):
        m = d == day
        goal = np.clip(tc[m][0], 0.5, m.sum() - 0.5)
        lo, hi = -20.0, 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(z[m] + mid)))).sum() < goal: lo = mid
            else: hi = mid
        out[m] = z[m] + strength * (lo + hi) / 2
    return 1 / (1 + np.exp(-out))
