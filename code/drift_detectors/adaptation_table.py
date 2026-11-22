"""Report table tab:adaptation with the current drift-sliding policy, January 2015 - October 2026.

The other rows of the table (yesterday, last week, frozen, expanding, two-year sliding) are unchanged and are read from
origin/price-c2:results/protocol/run_experiments/aggregate_metrics.csv; this script recomputes the drift-sliding row.

Models: the original Step 1 selections of run_experiments (price-c2 forecast_package, 240 features): demand LightGBM
63 leaves; price LightGBM 31 leaves with the cut-off chosen on 2014 by balanced accuracy (0.57, kept by every refit).

Drift sliding, current policy:
  input detector  daily value: weather drift score of day D (weather_drift.py, branch experiments): the day's 18 weather
                  summaries (temperature mean/max/min, humidity, cloud, solar; Sydney, Melbourne, Brisbane) minus a
                  least-squares 3-harmonic annual mean, divided by the square root of a 3-harmonic annual least-squares fit
                  to the squared deviations (floored at 5% of their mean), then the Mahalanobis distance (Ledoit-Wolf
                  covariance). Fitted on 2010-2013.
  error detector  daily value: the deployed model's demand MAPE (%) / price misclassification rate (%) on day D.
  both            statistic: 28-day rolling mean; at most one signal per 28 days (analysis_tools.OnlineDetector).
  thresholds      at the start: the 99.9th (input) / 99th (error) percentile of the 28-day means over 2014 (error: the
                  model fitted on 2010-2013, its 2014 errors). At every retrain both detectors are re-baselined on the
                  12 months up to the end of D, same percentiles: input - the weather scores of those 12 months; error -
                  the daily errors on those 12 months of the model fitted on the 3 years before them (out of sample).
  retraining      an input OR error signal on day D retrains on the 4 years up to the end of D, effective from D+1, if
                  there has been no retrain yet or at least 180 days have passed since the last one, and fewer than 6
                  retrains have been made (the rule of run_experiments.simulate_policy). A signal inside the 180 days
                  triggers nothing but still starts that detector's 28-day signal cooldown (as in simulate_policy).
Variants (results/drift_detectors_2015_2026/adaptation_table/):
  original        the old row: day-type input score, both thresholds at the 99th percentile of 2014, no re-baselining.
                  Must reproduce origin/price-c2 adapted_drift_sliding (a check of this harness).
  current         the policy above: the new table row.
  refit_scorer    as current, but every re-baseline also refits the weather scorer on the 3 years before the 12 months
                  (as plots/simulate_drift_rebaselined.py on branch experiments does).
  empty_window    as current, but the 28-day window starts empty at the start and after every re-baseline (the Detector
                  of simulate_drift_rebaselined.py) instead of holding the last 27 reference days.
  input_only      as current, but only the input detector's signals trigger retrains (both are still re-baselined).
  error_only      as current, but only the error detector's signals trigger retrains.
  no_cooldown     as current, but without the 28-day cooldown between signals: once the 180 days have passed, the first
                  day whose 28-day average exceeds a threshold triggers the retrain (the literal reading of the method).
Needs results/drift_detectors_2015_2026/cache_table.pkl (built by simulate_drift.py).
Run from the repo root: python code/drift_detectors/adaptation_table.py   (TABLE_ONLY=1: rebuild the table only;
ONLY=name1,name2: simulate only these variants, then rebuild the table)"""
import io, json, os, subprocess, sys, tempfile, time
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
from collections import deque
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); ROOT = os.path.dirname(CODE)
BASE = os.path.join(ROOT, "results", "drift_detectors_2015_2026")
SMOKE = bool(os.environ.get("SMOKE"))                              # SMOKE=1: 10-tree models, output in smoke/ (code check)
OUT = os.path.join(BASE, "adaptation_table", "smoke") if SMOKE else os.path.join(BASE, "adaptation_table")
CACHE = os.path.join(BASE, "cache_table.pkl")
CSV = os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")
OLD = "origin/price-c2:results/protocol/run_experiments/aggregate_metrics.csv"
OLD_RETRAINS = "origin/price-c2:results/protocol/run_experiments/retrains.csv"
CONFIG = {"demand": {"family": "boosted", "leaves": 63, "trees": 900, "learning_rate": 0.025},
          "price": {"family": "boosted", "leaves": 31, "trees": 600, "learning_rate": 0.04}}
WINDOW_YEARS, BUDGET, GAP_DAYS = 4, 6, 180
START, END, SPLIT = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-10-31"), pd.Timestamp("2024-12-31")
CURRENT = dict(inputs="weather", q_input=0.999, q_error=0.99, rebaseline=True, refit_scorer=False, seeded=True, cooldown=28,
               trigger="either")
VARIANTS = {
    "original": dict(inputs="day_type", q_input=0.99, q_error=0.99, rebaseline=False, refit_scorer=False, seeded=True,
                     cooldown=28, trigger="either"),
    "current": CURRENT,
    "input_only": dict(CURRENT, trigger="input"),
    "error_only": dict(CURRENT, trigger="error"),
    "refit_scorer": dict(CURRENT, refit_scorer=True),
    "empty_window": dict(CURRENT, seeded=False),
    "no_cooldown": dict(CURRENT, cooldown=0),
}
SOURCES = [("origin/price-c2", f"code/forecast_package/{f}") for f in ("model.py", "train.py", "random_forest.py")] + \
          [("7b3be21", "plots/weather_drift.py")]                       # Suhan's weather scorer (unchanged since)


def extract(dest):
    os.makedirs(dest, exist_ok=True)
    for ref, path in SOURCES:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout
        open(os.path.join(dest, os.path.basename(path)), "w").write(src)


def daily_error(task, actual, predicted):
    a, p = np.asarray(actual).reshape(-1, 48), np.asarray(predicted).reshape(-1, 48)
    return 100 * np.mean(np.abs(a - p) / a, axis=1) if task == "demand" else 100 * np.mean(a != p, axis=1)


def detector(reference, quantile, seeded, cooldown=28):
    """analysis_tools.OnlineDetector (window seeded with the last 27 reference days), or with an empty window."""
    from analysis_tools import OnlineDetector
    d = OnlineDetector(reference, quantile=quantile, cooldown=cooldown)
    if not seeded:
        d.values = deque(maxlen=d.window)
    return d


def simulate(job):
    task, variant, pkg = job
    sys.path[:0] = [pkg, CODE]
    from model import market_days, predict_task
    from train import fit_model
    from weather_drift import daily_weather, weather_scorer
    from sklearn.metrics import balanced_accuracy_score
    v, config = VARIANTS[variant], dict(CONFIG[task], **({"trees": 10} if SMOKE else {}))
    x, y, day_type_score = pd.read_pickle(CACHE)
    assert x.shape[1] == 240
    days = market_days(x.index)
    truth = y[task].to_numpy()
    uday = pd.DatetimeIndex(np.unique(np.asarray(days)))

    # Step 1: the model fitted on 2010-2013 gives the price cut-off and the initial error reference (its 2014 errors)
    fit13, v14 = np.asarray(days.year <= 2013), np.asarray(days.year == 2014)
    ref = fit_model(task, config, x[fit13], y[fit13])
    if task == "price":
        p = ref["estimator"].predict_proba(x.loc[v14, ref["features"]])[:, 1]
        grid = np.linspace(0.2, 0.8, 61)
        cut = float(grid[int(np.argmax([balanced_accuracy_score(truth[v14], p > t) for t in grid]))])
        assert SMOKE or abs(cut - 0.57) < 1e-9, cut
        config["threshold"] = ref["threshold"] = cut
    err14 = daily_error(task, truth[v14], predict_task(ref, x[v14]))

    if v["inputs"] == "weather":
        w = daily_weather(CSV)
        score = weather_scorer(w[w.index.year <= 2013])(w).reindex(uday)     # scorer fitted on 2010-2013
    else:
        w, score = None, day_type_score.reindex(uday)                       # team day types, fitted on 2010-2013
    det_in = detector(score[uday.year == 2014].to_numpy(), v["q_input"], v["seeded"], v["cooldown"])
    det_err = detector(err14, v["q_error"], v["seeded"], v["cooldown"])
    baselines = [{"effective_day": str(START.date()), "input_reference": "2014", "error_reference": "2014",
                  "error_reference_model": "2010-2013", "input_threshold": det_in.threshold,
                  "error_threshold": det_err.threshold}]

    def rebaseline(day, score):
        """Detectors for the end of D = day - 1: references = the 12 months up to the end of D."""
        D = day - pd.Timedelta(days=1)
        trail = (days > D - pd.DateOffset(years=1)) & (days <= D)
        fit3 = (days >= day - pd.DateOffset(years=WINDOW_YEARS)) & (days <= D - pd.DateOffset(years=1))
        trail, fit3 = np.asarray(trail), np.asarray(fit3)
        m3 = fit_model(task, config, x[fit3], y[fit3])
        err_ref = daily_error(task, truth[trail], predict_task(m3, x[trail]))
        tdays = (uday > D - pd.DateOffset(years=1)) & (uday <= D)
        if v["refit_scorer"]:
            fdays = uday[(uday >= day - pd.DateOffset(years=WINDOW_YEARS)) & (uday <= D - pd.DateOffset(years=1))]
            scorer = weather_scorer(w.loc[fdays])
            later = uday[uday > D - pd.DateOffset(years=1)]
            score = score.copy()
            score.loc[later] = scorer(w.loc[later]).to_numpy()
        info = {"effective_day": str(day.date()), "input_reference": f"{uday[tdays].min().date()}..{uday[tdays].max().date()}",
                "error_reference": f"{days[trail].min().date()}..{days[trail].max().date()}",
                "error_reference_model": f"{days[fit3].min().date()}..{days[fit3].max().date()}"}
        di = detector(score[tdays].to_numpy(), v["q_input"], v["seeded"], v["cooldown"])
        de = detector(err_ref, v["q_error"], v["seeded"], v["cooldown"])
        return di, de, score, dict(info, input_threshold=di.threshold, error_threshold=de.threshold)

    sim_days = pd.date_range(START, END, freq="D")
    sim = np.asarray((days >= START) & (days <= END))
    xs, actual = x[sim], truth[sim]
    assert len(xs) == 48 * len(sim_days) and (pd.DatetimeIndex(np.unique(np.asarray(days[sim]))) == sim_days).all()
    frozen = fit_model(task, config, x[np.asarray(days.year <= 2014)], y[np.asarray(days.year <= 2014)])
    frozen_pred = predict_task(frozen, xs)
    pred = frozen_pred.copy()
    last_fit, retrains, log, pending = pd.Timestamp("2014-12-31"), [], [], False
    for i, day in enumerate(sim_days):
        if pending and len(retrains) < BUDGET:                      # retrain (and re-baseline) effective from D+1
            t0 = time.time()
            win = np.asarray((days < day) & (days >= day - pd.DateOffset(years=WINDOW_YEARS)))
            model = fit_model(task, config, x[win], y[win])
            pred[i * 48:] = predict_task(model, xs.iloc[i * 48:])
            last_fit = day - pd.Timedelta(days=1)
            retrains.append({"effective_day": str(day.date()), "training_start": str(days[win].min().date()),
                             "training_end": str(days[win].max().date()), "training_rows": int(win.sum()),
                             "cause": log[-1]["cause"]})
            if v["rebaseline"]:
                det_in, det_err, score, info = rebaseline(day, score)
                baselines.append(info)
            retrains[-1]["seconds"] = round(time.time() - t0, 1)
        pending = False
        e = float(daily_error(task, actual[i * 48:(i + 1) * 48], pred[i * 48:(i + 1) * 48])[0])
        s_in = det_in.observe(day, float(score.loc[day])); s_err = det_err.observe(day, e)
        fired = {"either": s_in or s_err, "input": s_in, "error": s_err}[v["trigger"]]
        trigger = fired and len(retrains) < BUDGET and (not retrains or (day - last_fit).days >= GAP_DAYS)
        log.append({"day": str(day.date()), "error": e, "input_value": float(score.loc[day]),
                    "input_statistic": det_in.statistic, "input_threshold": det_in.threshold, "input_signal": bool(s_in),
                    "error_statistic": det_err.statistic, "error_threshold": det_err.threshold, "error_signal": bool(s_err),
                    "triggers_retrain": bool(trigger),
                    "cause": "+".join(n for n, s in (("input", s_in), ("error", s_err))
                                      if s and v["trigger"] in ("either", n)) if trigger else ""})
        if trigger:
            pending = True
    key = f"{variant}_{task}"
    np.save(os.path.join(OUT, f"{key}_forecast.npy"), pred)
    if variant == "original":
        np.save(os.path.join(OUT, f"frozen_{task}_forecast.npy"), frozen_pred)
        np.save(os.path.join(OUT, f"truth_{task}.npy"), actual)
    json.dump({"task": task, "variant": variant, "settings": v, "config": config, "retrains": retrains,
               "baselines": baselines, "log": log}, open(os.path.join(OUT, f"{key}.json"), "w"))
    print(f"{key}: {len(retrains)} retrains {[(r['effective_day'], r['cause']) for r in retrains]}", flush=True)
    return key


def table():
    sys.path[:0] = [CODE]
    from analysis_tools import metric_values
    old = pd.read_csv(io.StringIO(subprocess.run(["git", "-C", ROOT, "show", OLD], capture_output=True,
                                                           text=True, check=True).stdout))
    need = [f"truth_{k}.npy" for k in ("demand", "price")] + [f"{n}_{k}_forecast.npy" for n in ["frozen"] + list(VARIANTS)
                                                               for k in ("demand", "price")]
    missing = [f for f in need if not os.path.exists(os.path.join(OUT, f))]
    if missing:
        print("table skipped, still missing:", missing)
        return
    sim_days = pd.date_range(START, END, freq="D")
    periods = {"2015-2024": np.repeat(sim_days <= SPLIT, 48), "2025-Oct 2026": np.repeat(sim_days > SPLIT, 48)}
    t = {k: np.load(os.path.join(OUT, f"truth_{k}.npy")) for k in ("demand", "price")}
    rows = []
    for name in ["frozen"] + list(VARIANTS):
        f = {k: np.load(os.path.join(OUT, f"{name}_{k}_forecast.npy")) for k in ("demand", "price")}
        for p, m in periods.items():
            rows.append({"strategy": name, "period": p, **metric_values(t["demand"][m], t["price"][m],
                                                                        f["demand"][m], f["price"][m])})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "metrics_all_variants.csv"), index=False)

    # harness check: frozen and original must reproduce the saved run_experiments rows
    old_name = {"strategy_selection": "2015-2024", "holdout": "2025-Oct 2026"}
    check = []
    for mine, theirs in (("frozen", "adapted_frozen"), ("original", "adapted_drift_sliding")):
        for _, o in old[(old.model == theirs) & old.period.isin(old_name)].iterrows():
            r = R[(R.strategy == mine) & (R.period == old_name[o.period])].iloc[0]
            check.append({"row": mine, "period": old_name[o.period],
                          **{f"{c} diff": r[c] - o[c] for c in ("MAPE", "MAE", "accuracy", "balanced_accuracy")}})
    C = pd.DataFrame(check); C.to_csv(os.path.join(OUT, "reproduction_check.csv"), index=False)
    old_r = pd.read_csv(io.StringIO(subprocess.run(["git", "-C", ROOT, "show", OLD_RETRAINS],
                                                             capture_output=True, text=True, check=True).stdout))
    for task in ("demand", "price"):
        mine = [r["effective_day"] for r in json.load(open(os.path.join(OUT, f"original_{task}.json")))["retrains"]]
        theirs = old_r[(old_r.task == task) & (old_r.strategy == "drift_sliding")].effective_day.tolist()
        print(f"original {task} retrain dates match the saved run: {mine == theirs}  {mine}")

    # the table: other rows from the saved run, drift sliding from 'current'
    labels = [("Yesterday", "baseline_1d"), ("Last week", "baseline_7d"), ("Frozen", "adapted_frozen"),
              ("Expanding", "adapted_periodic_expanding"), ("Two-year sliding", "adapted_periodic_sliding"),
              (r"\quad Input detector", "input_only"), (r"\quad Error detector", "error_only"),
              (r"\quad Either detector", "current")]
    new = {"input_only", "error_only", "current"}
    cols = ("MAPE", "MAE", "accuracy", "balanced_accuracy")
    vals = {}
    for label, key in labels:
        for p_old, p in old_name.items():
            src = (R[(R.strategy == key) & (R.period == p)].iloc[0] if key in new
                   else old[(old.model == key) & (old.period == p_old)].iloc[0])
            vals[(label, p)] = [round(float(src[c]), 2) for c in cols]
    best = {}
    for p in old_name.values():
        for j, c in enumerate(cols):
            col = [vals[(label, p)][j] for label, _ in labels]
            best[(p, j)] = min(col) if c in ("MAPE", "MAE") else max(col)

    def cell(label, p, j):
        s = f"{vals[(label, p)][j]:.2f}"
        return f"\\textbf{{{s}}}" if vals[(label, p)][j] == best[(p, j)] else s
    lines = [r"\begin{table}[htbp]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.15}",
             r"\caption{Adaptation comparison using the original 240-feature LightGBMs: 63 leaves for demand, and 31 leaves "
             r"with threshold $0.57$ for price. Best result in each column is in bold.}\label{tab:adaptation}",
             r"\begin{tabular}{@{}lrrrr@{\hspace{1.5em}}rrrr@{}}", r"\toprule",
             r"& \multicolumn{4}{c}{\textbf{2015-2024}} & \multicolumn{4}{c}{\textbf{2025-Oct.\ 2026}} \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
             r"\textbf{Strategy} & MAPE (\%) & MAE (MW) & Acc.\ (\%) & BA (\%) & MAPE (\%) & MAE (MW) & Acc.\ (\%) & BA (\%) \\",
             r"\midrule"]
    for label, _ in labels:
        cells = " & ".join(cell(label, p, j) for p in old_name.values() for j in range(4))
        lines.append(f"{label:<23} & {cells} \\\\")
        if label == "Last week":
            lines.append(r"\midrule")
        if label == "Two-year sliding":
            lines += [r"\midrule", r"\multicolumn{9}{@{}l}{\textit{Drift sliding, retrain triggered by:}} \\"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "table_adaptation.tex"), "w").write("\n".join(lines) + "\n")
    pd.set_option("display.width", 200)
    print(C.round(4).to_string(index=False)); print(R.round(3).to_string(index=False)); print("\n".join(lines))


def paired(draws=2000, seed=9417):
    """Paired day-block bootstrap of each new drift policy against Expanding (the best strategy on 2015-2024):
    resample whole days (same days for both strategies); CI of the difference policy - expanding."""
    preds = os.path.join(tempfile.mkdtemp(prefix="adaptation_"), "p.parquet")
    with open(preds, "wb") as fh:
        fh.write(subprocess.run(["git", "-C", ROOT, "show", OLD.replace("aggregate_metrics.csv", "half_hour_predictions.parquet")],
                                capture_output=True, check=True).stdout)
    saved = pd.read_parquet(preds)
    t = {k: np.load(os.path.join(OUT, f"truth_{k}.npy")).reshape(-1, 48) for k in ("demand", "price")}
    assert (saved["nsw_demand"].to_numpy().reshape(-1, 48) == t["demand"]).all()

    def per_day(d, p):        # daily sums: APE, AE, correct, correct positives, positives, correct negatives, negatives
        d, p = d.reshape(-1, 48), p.reshape(-1, 48)
        pos = t["price"] == 1
        return np.column_stack([(np.abs(t["demand"] - d) / t["demand"]).sum(1) * 100, np.abs(t["demand"] - d).sum(1),
                                (p == t["price"]).sum(1) * 100, ((p == 1) & pos).sum(1), pos.sum(1),
                                ((p == 0) & ~pos).sum(1), (~pos).sum(1)])

    def metrics(s):           # s: summed daily columns over a sample of days (rows = draws)
        n = s[..., 4] + s[..., 6]
        return np.stack([s[..., 0] / n, s[..., 1] / n, s[..., 2] / n,
                         50 * (s[..., 3] / s[..., 4] + s[..., 5] / s[..., 6])], axis=-1)
    base = per_day(saved["adapted_periodic_expanding_demand"].to_numpy(), saved["adapted_periodic_expanding_price"].to_numpy())
    sim_days = pd.date_range(START, END, freq="D")
    rng = np.random.default_rng(seed)
    rows = []
    for p, m in (("2015-2024", sim_days <= SPLIT), ("2025-Oct 2026", sim_days > SPLIT)):
        idx = np.flatnonzero(m)
        boot = idx[rng.integers(0, len(idx), (draws, len(idx)))]
        for name in ("input_only", "error_only", "current"):
            mine = per_day(np.load(os.path.join(OUT, f"{name}_demand_forecast.npy")),
                           np.load(os.path.join(OUT, f"{name}_price_forecast.npy")))
            diff = metrics(mine[idx].sum(0)) - metrics(base[idx].sum(0))
            bd = metrics(mine[boot].sum(1)) - metrics(base[boot].sum(1))
            lo, hi = np.percentile(bd, [2.5, 97.5], axis=0)
            for j, c in enumerate(("MAPE", "MAE", "accuracy", "balanced_accuracy")):
                rows.append({"policy": name, "period": p, "metric": c, "policy minus expanding": diff[j],
                             "ci_low": lo[j], "ci_high": hi[j], "excludes 0": bool(lo[j] > 0 or hi[j] < 0)})
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, "paired_vs_expanding.csv"), index=False)
    print(P.round(3).to_string(index=False))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    if not os.environ.get("TABLE_ONLY"):
        pkg = tempfile.mkdtemp(prefix="adaptation_pkg_"); extract(pkg)
        only = [n for n in os.environ.get("ONLY", "").split(",") if n] or list(VARIANTS)
        jobs = [(task, variant, pkg) for task in ("demand", "price") for variant in VARIANTS if variant in only]
        with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as pool:
            for key in pool.map(simulate, jobs):
                print("done", key, flush=True)
    table()
    if all(os.path.exists(os.path.join(OUT, f"{n}_price_forecast.npy")) for n in VARIANTS):
        paired()
