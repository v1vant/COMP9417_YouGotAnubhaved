"""Online drift detectors for every original Step 1 model, with drift-triggered retraining, January 2015 - October 2026.

Models (Step 1 candidates, best per family on 2014; price cut-off fixed at 0.5):
  demand LightGBM 63 leaves, demand Ridge alpha=1000, price LightGBM 31 leaves, price Logistic C=0.01.
Step 1 code with the original 240 features: code/forecast_package of branch price-c2 (extracted at run time).
Detectors: analysis_tools.OnlineDetector (the team implementation), exactly as in run_experiments.simulate_policy:
  input detector  - daily value: weather drift score of day D (weather_drift.py: Mahalanobis distance of D's
                    seasonally-adjusted temperature / humidity / cloud / solar summaries for Sydney, Melbourne and
                    Brisbane, climatology and covariance fitted on 2010-2013); reference: the 2014 daily scores.
  error detector  - daily value: the deployed model's error on day D (demand: MAPE %, price: % of half-hours wrong);
                    reference: the daily 2014 errors of the same model fitted on 2010-2013 (its Step 1 validation year).
  both            - statistic: 28-day rolling mean; threshold: 99th percentile of the reference's 28-day rolling means;
                    at least 28 days between signals.
Retraining (run_experiments 'drift_sliding'): day D is forecast, then observed at the end of D; a triggered retrain takes
effect from D+1 and uses the 4 years before D+1 (data up to the end of D). Budget (spec): at most 6 retrains per model and
strategy between January 2015 and October 2026. No minimum gap between retrains (the spec sets none).
Two separate policies per model: retraining triggered only by the input detector, and only by the error detector.
Both detectors are logged every day in both policies. Run from the repo root: python plots/simulate_drift.py"""
import json, os, subprocess, sys, tempfile, time
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); CODE = os.path.join(ROOT, "code")
OUT = os.path.join(ROOT, "results", "drift_detectors_2015_2026")
CSV = os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")
MODELS = {
    "demand_lightgbm63": ("demand", {"family": "boosted", "leaves": 63, "trees": 900, "learning_rate": 0.025}),
    "demand_ridge1000": ("demand", {"family": "linear", "regularization": 1000.0}),
    "price_lightgbm31": ("price", {"family": "boosted", "leaves": 31, "trees": 600, "learning_rate": 0.04}),
    "price_logistic001": ("price", {"family": "linear", "regularization": 0.01}),
}
CUTOFF, WINDOW_YEARS, BUDGET = 0.5, 4, 6
START, END = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-10-31")


def step1_package():
    d = os.path.join(tempfile.gettempdir(), "step1_pkg_price_c2")
    os.makedirs(d, exist_ok=True)
    for f in ("model.py", "train.py", "random_forest.py"):     # atomic write: parallel workers may import meanwhile
        src = subprocess.run(["git", "-C", ROOT, "show", f"origin/price-c2:code/forecast_package/{f}"],
                             capture_output=True, text=True, check=True).stdout
        tmp = os.path.join(d, f"{f}.{os.getpid()}.tmp")
        open(tmp, "w").write(src)
        os.replace(tmp, os.path.join(d, f))
    sys.path[:0] = [d, CODE, HERE]


def data():
    cache = os.path.join(OUT, "cache_table.pkl")
    if not os.path.exists(cache):
        from model import load_data, training_table
        x, y = training_table(load_data(CSV))
        assert x.shape[1] == 240
        pd.to_pickle((x, y), cache)
    return pd.read_pickle(cache)[:2]                          # older caches also hold a third (unused) item


def weather_scores(fit_end_year=2013):
    """Daily weather drift scores for every market day, scorer fitted on 2010..fit_end_year (fixed)."""
    from weather_drift import daily_weather, weather_scorer
    w = daily_weather(CSV)
    return weather_scorer(w[w.index.year <= fit_end_year])(w)


def predict(task, model, rows):
    if task == "demand":
        from model import predict_task
        return predict_task(model, rows)
    return (model["estimator"].predict_proba(rows.loc[:, model["features"]])[:, 1] > CUTOFF).astype(int)


def daily_error(task, actual, predicted):
    a, p = actual.reshape(-1, 48), predicted.reshape(-1, 48)
    return 100 * np.mean(np.abs(a - p) / a, axis=1) if task == "demand" else 100 * np.mean(a != p, axis=1)


def run_model(key):
    step1_package()
    from model import market_days
    from train import fit_model
    from analysis_tools import OnlineDetector
    task, config = MODELS[key]
    x, y = data()
    score = weather_scores()                                         # weather scorer fitted on 2010-2013
    days = market_days(x.index)
    col = "demand" if task == "demand" else "price"
    # error-detector reference: the model fitted on 2010-2013, its daily errors on 2014
    ref_model = fit_model(task, config, x[days.year <= 2013], y[days.year <= 2013])
    v = np.asarray(days.year == 2014)
    err_ref = daily_error(task, y[col].to_numpy()[v], predict(task, ref_model, x[v]))
    inp_ref = score[score.index.year == 2014].to_numpy()
    frozen = fit_model(task, config, x[days.year <= 2014], y[days.year <= 2014])       # the Step 1 frozen model
    sim_days = pd.date_range(START, END, freq="D")
    rows = np.asarray((days >= START) & (days <= END))
    xs, actual = x[rows], y[col].to_numpy()[rows]
    frozen_pred = predict(task, frozen, xs)
    out = {"task": task, "config": config, "error_reference_daily": err_ref.tolist(),
           "frozen_daily_error": daily_error(task, actual, frozen_pred).tolist()}
    for trigger in ("input", "error"):
        model, pred = frozen, frozen_pred.copy()
        det_in, det_err = OnlineDetector(inp_ref), OnlineDetector(err_ref)
        last_fit, retrains, log, pending = pd.Timestamp("2014-12-31"), [], [], False
        for i, day in enumerate(sim_days):
            if pending and len(retrains) < BUDGET:                   # takes effect from this day (D+1)
                mask = (days < day) & (days >= day - pd.DateOffset(years=WINDOW_YEARS))
                t0 = time.time()
                model = fit_model(task, config, x[mask], y[mask])
                pred[i * 48:] = predict(task, model, xs.iloc[i * 48:])
                last_fit = day - pd.Timedelta(days=1)
                retrains.append({"effective_day": str(day.date()), "training_start": str(days[mask].min().date()),
                                 "training_end": str(days[mask].max().date()), "training_rows": int(mask.sum()),
                                 "seconds": round(time.time() - t0, 1)})
            pending = False
            e = float(daily_error(task, actual[i * 48:(i + 1) * 48], pred[i * 48:(i + 1) * 48])[0])
            s_in = det_in.observe(day, float(score.loc[day])); s_err = det_err.observe(day, e)
            log.append({"day": str(day.date()), "error": e, "input_statistic": det_in.statistic, "input_signal": s_in,
                        "error_statistic": det_err.statistic, "error_signal": s_err})
            fire = s_in if trigger == "input" else s_err
            if fire and len(retrains) < BUDGET:
                pending = True
        out[trigger] = {"input_threshold": det_in.threshold, "error_threshold": det_err.threshold,
                        "retrains": retrains, "log": log}
        print(f"{key} [{trigger}-triggered]: {len(retrains)} retrains: {[r['effective_day'] for r in retrains]}", flush=True)
    json.dump(out, open(os.path.join(OUT, f"{key}.json"), "w"))
    return key


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    step1_package(); data()                                          # build the shared cache once
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as pool:
        for k in pool.map(run_model, MODELS):
            print("done", k, flush=True)
