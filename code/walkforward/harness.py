"""Experiment harness (pure test; not part of the repo).
Data: a truncated copy of the CSV holding only rows up to 2025-01-01 00:00 (the last interval of
market day 2024-12-31). The 2025-2026 holdout is never read.
Schedule (as the team pipeline, run_experiments.simulate_policy 'periodic_expanding'):
  fit on <= 2013 -> predict 2014 (tuning year only)
  fit on <= 2014 -> predict 2015;  <= 2015 -> 2016-2017;  <= 2017 -> 2018-2019;
  <= 2019 -> 2020-2021;  <= 2021 -> 2022-2023;  <= 2023 -> 2024."""
import json, os, sys, time
import numpy as np, pandas as pd
CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(CODE, "..", ".."))
E = os.path.join(REPO, "results", "walkforward", "cache")      # data cache (git-ignored), rebuilt by build_cache()
RUNS = os.environ.get("WALKFORWARD_RUNS", os.path.join(REPO, "results", "walkforward", "runs"))   # saved predictions
sys.path.insert(0, f"{REPO}/code/forecast_package")
SCHEDULE = [(2013, [2014]), (2014, [2015]), (2015, [2016, 2017]), (2017, [2018, 2019]),
            (2019, [2020, 2021]), (2021, [2022, 2023]), (2023, [2024])]
N_ROWS = 5479 * 48        # 2010-01-01 00:30 .. 2025-01-01 00:00
DEMAND_CFG = {"family": "boosted", "leaves": 63, "trees": 900, "learning_rate": 0.025}


def build_cache():
    from model import load_data, training_table, market_days
    raw = pd.read_csv(f"{REPO}/data/nem_weather_halfhourly_2010_2026.csv", nrows=N_ROWS)
    assert raw["timestamp"].iloc[0] == "2010-01-01 00:30:00" and raw["timestamp"].iloc[-1] == "2025-01-01 00:00:00"
    os.makedirs(E, exist_ok=True)
    raw.to_csv(f"{E}/data_upto_2024.csv", index=False)
    features, truth = training_table(load_data(f"{E}/data_upto_2024.csv"))
    days = market_days(features.index)
    assert days.max() == pd.Timestamp("2024-12-31")
    np.save(f"{E}/X.npy", features.to_numpy(np.float32))
    np.save(f"{E}/index.npy", features.index.to_numpy())
    truth.to_parquet(f"{E}/truth.parquet")
    json.dump(list(features.columns), open(f"{E}/columns.json", "w"))
    print("cache:", features.shape, days.min().date(), days.max().date())


def load():
    from model import market_days
    idx = pd.DatetimeIndex(np.load(f"{E}/index.npy", allow_pickle=True))
    X = pd.DataFrame(np.load(f"{E}/X.npy"), index=idx, columns=json.load(open(f"{E}/columns.json")))
    truth = pd.read_parquet(f"{E}/truth.parquet")
    days = market_days(idx)
    assert days.max() == pd.Timestamp("2024-12-31")
    return X, truth, days


# ---------- demand forecasts as price features ----------
def demand_forecast_job(year):
    """Demand LightGBM trained on years < year, forecasting year (what was knowable then)."""
    path = f"{RUNS}/demandfc__{year}.npy"
    if os.path.exists(path):
        return path
    from model import predict_task
    from train import fit_model
    X, truth, days = load()
    t0 = time.time()
    model = fit_model("demand", DEMAND_CFG, X[days.year < year], truth[days.year < year])
    np.save(path, predict_task(model, X[days.year == year]).astype(np.float32))
    json.dump({"seconds": time.time() - t0}, open(path + ".json", "w"))
    return path


# Schedule-matched demand forecasts: in the years where the price model is a year old (2017, 2019,
# 2021, 2023), the demand model used for the forecast is equally old (trained on years <= year - 2),
# so the demand input cannot carry fresher information than the price model's own refit schedule.
SCHED_STALE = {2017: 2015, 2019: 2017, 2021: 2019, 2023: 2021}


def demand_forecast_sched_job(year):
    path = f"{RUNS}/demandfc_sched__{year}.npy"
    if os.path.exists(path):
        return path
    from model import predict_task
    from train import fit_model
    X, truth, days = load()
    t0 = time.time()
    fit = days.year <= SCHED_STALE[year]
    model = fit_model("demand", DEMAND_CFG, X[fit], truth[fit])
    np.save(path, predict_task(model, X[days.year == year]).astype(np.float32))
    json.dump({"seconds": time.time() - t0, "fit_end": SCHED_STALE[year]}, open(path + ".json", "w"))
    return path


def demand_forecast(days, schedule_matched=False):
    fc = np.full(len(days), np.nan, np.float32)
    for year in range(2011, 2025):
        name = "demandfc_sched" if schedule_matched and year in SCHED_STALE else "demandfc"
        fc[np.asarray(days.year == year)] = np.load(f"{RUNS}/{name}__{year}.npy")
    return fc


def add_demand_features(X, truth, days, variant, schedule_matched=False, source=None):
    """D1 point; D2 + relative to recent actuals; D3 + whole-day summaries of tomorrow's forecast;
    D4 + tomorrow's 48-slot forecast profile; D5 = D3 built from ACTUAL demand (ceiling, not deployable)."""
    if source and source.endswith("_blend"):   # anubhav_blend: his 0.5 LightGBM + 0.5 MLP forecast (build_anubhav_blend.py);
        f = np.load(f"{RUNS}/{source}_forecast.npy")   # solar_blend / ctrl_blend: solar/run_fixes.py
    else:
        f = truth["demand"].to_numpy(np.float64) if variant == "D5" else demand_forecast(days, schedule_matched).astype(np.float64)
    new = {"dfc": f}
    if variant == "A5":   # Anubhav's five features: test_forecast_demand_price.added_features (commit 4f32442)
        new.update(dfc_minus_lag1=f - X["nsw_demand_lag_1"], dfc_minus_lag7=f - X["nsw_demand_lag_7"],
                   dfc_minus_prev_mean=f - X["nsw_demand_previous_mean"],
                   dfc_minus_prev_max=f - X["nsw_demand_previous_max"])
    if variant in ("D2", "D3", "D4", "D5"):
        new.update(dfc_minus_anchor=f - X["demand_anchor"], dfc_minus_lag1=f - X["nsw_demand_lag_1"],
                   dfc_minus_lag7=f - X["nsw_demand_lag_7"], dfc_ratio_lag1=f / X["nsw_demand_lag_1"])
    if variant in ("D3", "D4", "D5"):
        g = pd.Series(f, index=X.index).groupby(np.asarray(days))
        mean, mx, mn, sd = (g.transform(s).to_numpy() for s in ("mean", "max", "min", "std"))
        prev = pd.Series(f, index=X.index).groupby(np.asarray(days)).shift(1)
        nxt = pd.Series(f, index=X.index).groupby(np.asarray(days)).shift(-1)
        new.update(dfc_day_mean=mean, dfc_day_max=mx, dfc_day_min=mn, dfc_day_std=sd,
                   dfc_day_mean_minus_prev_mean=mean - X["nsw_demand_previous_mean"],
                   dfc_minus_day_mean=f - mean, dfc_minus_day_max=f - mx,
                   dfc_rank_in_day=g.rank(pct=True).to_numpy(),
                   dfc_ramp_prev=(pd.Series(f, index=X.index) - prev).fillna(0).to_numpy(),
                   dfc_ramp_next=(nxt - pd.Series(f, index=X.index)).fillna(0).to_numpy())
    if variant == "D4":
        slot = X["slot"].to_numpy().astype(int)
        prof = pd.DataFrame({"day": np.asarray(days), "slot": slot, "f": f}).pivot(index="day", columns="slot", values="f")
        dmap = pd.Series(np.arange(len(prof)), index=prof.index)
        rows = dmap.loc[np.asarray(days)].to_numpy()
        base = X["nsw_demand_previous_mean"].to_numpy()
        for s in range(48):
            new[f"dfc_profile_slot_{s}"] = prof[s].to_numpy()[rows] - base
    out = X.copy()
    for k, v in new.items():
        out[k] = np.asarray(v, dtype=np.float32)
    return out


def oof_series(run, early, days, key="y"):
    """Walk-forward out-of-sample values of a saved run for every row from 2011 (see lgbm_logit_series).
    key 'y' -> logit of the saved probability; key 'r' -> the saved raw score (e.g. a regression's prediction)."""
    v = np.full(len(days), np.nan)
    years = np.asarray(days.year)
    for fit_end, ys in [(2010, [2011]), (2011, [2012]), (2012, [2013])] + SCHEDULE:
        z = np.load(f"{RUNS}/{early if fit_end < 2013 else run}__{fit_end}.npz")
        for y in ys:
            m = years == y
            assert m.sum() == len(z[f"{key}{y}"])
            v[m] = z[f"{key}{y}"]
    if key == "y":
        v = np.clip(v, 1e-6, 1 - 1e-6); v = np.log(v / (1 - v))
    return v


def lgbm_logit_series(run, early, days):
    """Logit of LightGBM's probability for every row, each from a model fitted only on earlier years:
    2011-2013 from the extra fits {early}__2010/2011/2012, 2014-2024 from {run} on the team SCHEDULE.
    The values for a predicted year are exactly what the deployed LightGBM (refitted at the same
    fit_end) would output, and every training-row value is out-of-sample."""
    p = np.full(len(days), np.nan)
    years = np.asarray(days.year)
    for fit_end, ys in [(2010, [2011]), (2011, [2012]), (2012, [2013])] + SCHEDULE:
        z = np.load(f"{RUNS}/{early if fit_end < 2013 else run}__{fit_end}.npz")
        for y in ys:
            m = years == y
            assert m.sum() == len(z[f"y{y}"])
            p[m] = z[f"y{y}"]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------- one fit of one configuration ----------
def fit_job(name, spec, fit_end, predict_years):
    path = f"{RUNS}/{name}__{fit_end}.npz"
    if os.path.exists(path):
        return path
    X, truth, days = load()
    start = spec.get("start_year", 2010)
    late = offset = None
    if spec.get("demand"):
        Xd = add_demand_features(X, truth, days, spec["demand"], spec.get("demand_schedule_matched", False),
                                 spec.get("demand_source"))
        if spec.get("demand_mode") == "late":      # demand forecasts join the last hidden layer
            late = Xd[[c for c in Xd.columns if c not in X.columns]].to_numpy(np.float64)
        else:
            X = Xd
        del Xd
    if spec.get("weather_shape"):                # Suhan's 26 Sydney weather-shape inputs, appended after the base 571
        X = X.copy()
        for k, name in enumerate(json.load(open(f"{RUNS}/weather_shape_names.json"))):
            X[name] = np.load(f"{RUNS}/weather_shape.npy", mmap_mode="r")[:, k]
    if spec.get("neg_features"):                 # explicit negative-price features (day D and earlier; build_negative_features.py)
        X = X.copy()
        for k, name in enumerate(json.load(open(f"{RUNS}/negprice_names.json"))):
            X[name] = np.load(f"{RUNS}/negprice.npy", mmap_mode="r")[:, k]
    for f in spec.get("oof_features", []):        # out-of-sample outputs of earlier models as extra inputs
        X = X.copy(); X[f["name"]] = oof_series(f["run"], f["early"], days, f.get("key", "y")).astype(np.float32)
    lf = spec.get("layered_features")             # out-of-sample layered outputs as extra classifier inputs
    if lf:
        import layered
        mu, q10, q50, q90, lv = (oof_series(lf["day"], lf["day_early"], days, k) for k in ("n", "q10", "q50", "q90", "l"))
        sc = oof_series(lf["intra"], lf["intra_early"], days, "r")
        P, r = layered.prob_from_count(sc, mu, q10, q90, days)
        X = X.copy()
        for k, v in dict(lay_n=mu, lay_q10=q10, lay_q50=q50, lay_q90=q90, lay_level=lv, intra_score=sc,
                         intra_rank=r, lay_p=P).items():
            X[k] = v.astype(np.float32)
    if spec.get("lgbm"):                           # out-of-sample LightGBM logit as an input or output neuron
        g = lgbm_logit_series(spec["lgbm"]["run"], spec["lgbm"]["early"], days)
        if spec["lgbm"]["mode"] == "input":
            X = X.copy(); X["lgbm_logit"] = g.astype(np.float32)
        else:
            offset = g
    train = np.asarray((days.year <= fit_end) & (days.year >= start))
    pred = np.asarray(days.year.isin(predict_years))
    t0 = time.time()
    extras = {}
    if spec.get("top_features"):
        # Keep only the inputs with the largest LightGBM gain on THIS training window (a 300-tree probe).
        import knn
        cols = list(X.columns)
        gain = knn.gain_weights(X, truth["price"].to_numpy().astype(np.int8), train, cols)
        X = X[[cols[i] for i in np.argsort(gain)[::-1][:spec["top_features"]]]]
    if spec.get("recency_tau"):
        # LightGBM-classifier exactly as train.fit_model's price branch, plus sample weights that decay
        # with the age of each training row: w = exp(-age_years / tau), measured from the window's end.
        from lightgbm import LGBMClassifier
        c = spec["config"]
        est = LGBMClassifier(objective="binary", n_estimators=c["trees"], learning_rate=c["learning_rate"],
                             num_leaves=c["leaves"], min_child_samples=160, colsample_bytree=0.85, subsample=0.85,
                             subsample_freq=1, reg_lambda=8.0, max_bin=127, n_jobs=2, verbosity=-1,
                             deterministic=True, force_col_wise=True, random_state=9417, class_weight="balanced")
        end = pd.Timestamp(f"{fit_end + 1}-01-01")
        age = np.asarray((end - X.index[train]) / pd.Timedelta(days=365.25), dtype=np.float64)
        w = np.exp(-age / spec["recency_tau"]); w /= w.mean()
        est.fit(X[train], truth["price"].to_numpy()[train], sample_weight=w)
        prob = np.concatenate([est.predict_proba(X[pred].iloc[i:i + 20000])[:, 1] for i in range(0, int(pred.sum()), 20000)])
    elif spec["kind"] == "torch":
        import subprocess, tempfile
        tmp = tempfile.mkdtemp(dir=E)
        Xp = X
        np.save(f"{tmp}/X.npy", Xp.to_numpy(np.float32))
        json.dump(list(Xp.columns), open(f"{tmp}/cols.json", "w"))
        np.save(f"{tmp}/train.npy", train); np.save(f"{tmp}/pred.npy", pred)
        np.save(f"{tmp}/y.npy", truth["price"].to_numpy().astype(np.float32))
        np.save(f"{tmp}/years.npy", np.asarray(days.year))           # for inner (within-window) early stopping
        np.save(f"{tmp}/day_codes.npy", pd.factorize(np.asarray(days))[0])
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
        subprocess.run([sys.executable, f"{CODE}/{spec.get('script', 'torch_alt.py')}", tmp, json.dumps(spec["config"])],
                       check=True, env=env)
        prob = np.load(f"{tmp}/prob.npy")
        import shutil; shutil.rmtree(tmp)
    elif spec["kind"] == "knn":
        import knn
        cfg, y = spec["config"], truth["price"].to_numpy().astype(np.int8)
        if cfg["kind"] == "analog":
            prob = knn.analog_vote(knn.analog_setup(X, y, days, train, pred, cfg["k"]), cfg["k"], cfg["vote"])
        else:
            R, _ = knn.representation(X, y, days, train, cfg["kind"], cfg["dims"], cfg["p"], np.random.default_rng(0))
            metric = {2: "euclidean", 1: "manhattan", "cosine": "cosine"}[cfg["p"]]
            dist, idx = knn.neighbours(R, train, pred, cfg["k"], metric)
            years_f = np.asarray((X.index - pd.Timestamp("2010-01-01")) / pd.Timedelta(days=365.25), dtype=np.float64)
            age = years_f[pred][:, None] - years_f[train][idx]
            prob = knn.vote(dist, idx, y[train].astype(np.float64), cfg["k"], cfg["vote"], age, cfg.get("tau"))
    elif spec["kind"] == "lgbx":
        import lgbx
        tg = np.load(f"{RUNS}/targets.npz")
        oofp = None
        if spec.get("weights") == "oof_uncertainty":
            g = lgbm_logit_series("lgbm63", "lgbm63_early", days); oofp = 1 / (1 + np.exp(-g))
        target = None
        if spec.get("target") == "given":
            if spec.get("given_target", "rank") == "rank":   # within-day percentile rank of tomorrow's price
                target = pd.Series(tg["price"]).groupby(np.asarray(days)).rank(pct=True).to_numpy()[train]
            elif spec["given_target"] == "intra_rel":       # deviation from tomorrow's own mean, relative to today's mean
                target = np.clip((tg["price"] - tg["mean_next"]) / np.maximum(np.abs(tg["thr"]), 10.0), -3, 3)[train]
        if spec.get("task") == "lambdarank":                # rank slots within each day (query = market day)
            pct = pd.Series(tg["price"]).groupby(np.asarray(days)).rank(pct=True).to_numpy()
            lab = np.minimum((pct * 10).astype(int), 9)[train]
            dtr = np.asarray(days)[train]
            groups = pd.Series(dtr).groupby(dtr, sort=False).size().to_numpy()
            assert (np.diff(pd.factorize(dtr)[0]) >= 0).all(), "training rows must be grouped by day"
            prob, raw = lgbx.rank_fit_predict(spec, X[train], X[pred], lab, groups)
        else:
            prob, raw = lgbx.fit_predict(spec, X[train], X[pred], truth["price"].to_numpy()[train],
                                         tg["margin"][train], tg["thr"][train], None if oofp is None else oofp[train], target)
        if raw is not None:
            extras["r"] = raw
    elif spec["kind"] == "day":
        import layered
        tg = np.load(f"{RUNS}/targets.npz")
        d = np.asarray(days)
        T = layered.day_table(X, days)
        n_day = pd.Series(truth["price"].to_numpy(), index=d).groupby(level=0).sum()
        lvl = pd.Series(tg["mean_next"] - tg["thr"], index=d).groupby(level=0).first()
        tdays = T.index[(T.index.year >= start) & (T.index.year <= fit_end)]
        pdays = T.index[T.index.year.isin(predict_years)]
        lrel = None
        if spec.get("level_rel"):                  # relative level target: (mean(D+1) - mean(D)) / max(|mean(D)|, 10)
            thr_day = pd.Series(tg["thr"], index=d).groupby(level=0).first()
            lrel = (lvl / np.maximum(np.abs(thr_day), 10.0)).clip(-3, 3)
        Dp = layered.fit_predict_day(T, n_day, lvl, tdays, pdays, tuple(spec.get("quantiles", (0.1, 0.5, 0.9))), lrel)
        rows = d[pred]
        prob = np.clip(Dp["n"].reindex(rows).to_numpy() / 48, 0, 1)
        extras = {k: Dp[k].reindex(rows).to_numpy() for k in Dp.columns}
    elif spec["kind"] == "nnx":
        import nn_ext
        sl = lambda v, m: None if v is None else v[m]
        params = nn_ext.fit(spec["config"], X[train], truth["price"][train], sl(late, train), sl(offset, train),
                            spec.get("lgbm", {}).get("learn", True), spec.get("lgbm", {}).get("a_lr_mult", 1.0),
                            spec.get("z_clip"))
        prob = np.concatenate([nn_ext.predict(params, X[pred].iloc[i:i + 20000],
                                              sl(late, pred)[i:i + 20000] if late is not None else None,
                                              sl(offset, pred)[i:i + 20000] if offset is not None else None)
                               for i in range(0, int(pred.sum()), 20000)])
        if offset is not None:
            json.dump({"offset_scales": params["offset_scales"]}, open(path + ".scales.json", "w"))
    else:
        from model import price_probability
        from train import fit_model
        model = fit_model("price", spec["config"], X[train], truth[train])
        prob = np.concatenate([price_probability(model, X[pred].iloc[i:i + 20000])
                               for i in range(0, int(pred.sum()), 20000)])
    seconds = time.time() - t0
    years = np.asarray(days.year[pred])
    out = {f"y{y}": prob[years == y].astype(np.float32) for y in predict_years}
    for k, v in extras.items():
        out.update({f"{k}{y}": np.asarray(v)[years == y].astype(np.float32) for y in predict_years})
    np.savez(path, **out)
    import hashlib
    code = hashlib.sha256(b"".join(open(f"{CODE}/{f}", "rb").read() for f in ("harness.py", "lgbx.py", "layered.py", "nn_ext.py")
                                   if os.path.exists(f"{CODE}/{f}"))).hexdigest()[:12]
    json.dump({"seconds": seconds, "train_rows": int(train.sum()), "fit_end": fit_end, "code_sha": code, "spec": spec},
              open(path + ".json", "w"))
    return path
