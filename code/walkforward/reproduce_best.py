"""Reproduce the best-so-far price-direction configuration on the team walk-forward (2015-2024, threshold 0.5).

Best so far: 0.7 * P(LightGBM-63 + Anubhav's 5 demand-forecast features) + 0.3 * P(NN 256x256 + D4 demand features),
both models using Anubhav's day-ahead demand forecast (0.5 LightGBM + 0.5 MLP) and both trained from 2011.

Usage (from the repo root, with the project venv):
    python code/walkforward/reproduce_best.py                 # score the shipped predictions (a few minutes)
    python code/walkforward/reproduce_best.py --refit lgbm63_A5_blend nn256_D4_blend --fit-ends 2013
        # refit those configurations for the listed walk-forward fits into results/walkforward/refit/
        # and compare with the shipped predictions (full refit of everything: about 1-2 hours on 4 cores)

Data: only the first 5479*48 rows of data/nem_weather_halfhourly_2010_2026.csv are read (to 2025-01-01 00:00, the last
interval of market day 2024-12-31). Nothing from the 2025-2026 holdout is read, trained on or scored.
"""
import argparse, json, os, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "code", "forecast_package"), HERE]
os.environ["FIXED_THRESHOLD"] = "1"
for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")

SHIPPED = os.path.join(REPO, "results", "walkforward", "runs")
BEST = [("lgbm63_A5_blend", 0.7), ("nn256_D4_blend", 0.3)]


def _fit(args):
    name, spec, fit_end, years, runs = args
    os.environ["WALKFORWARD_RUNS"] = runs
    import harness
    harness.RUNS = runs
    return harness.fit_job(name, spec, fit_end, years)


def refit(names, fit_ends, workers):
    import numpy as np
    import harness
    runs = os.path.join(REPO, "results", "walkforward", "refit")
    os.makedirs(runs, exist_ok=True)
    shutil.copy(os.path.join(SHIPPED, "anubhav_blend_forecast.npy"), runs)
    specs = json.load(open(os.path.join(SHIPPED, "configs.json")))
    jobs = [(n, specs[n], fe, ys, runs) for n in names for fe, ys in harness.SCHEDULE if not fit_ends or fe in fit_ends]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_fit, jobs))
    print("\nRefit vs shipped predictions (max |probability difference|):")
    for n, _, fe, ys, _ in jobs:
        a, b = np.load(f"{runs}/{n}__{fe}.npz"), np.load(f"{SHIPPED}/{n}__{fe}.npz")
        print(f"  {n:18s} fit<={fe}: " + ", ".join(f"{y}: {float(np.max(np.abs(a[f'y{y}'] - b[f'y{y}']))):.2e}" for y in ys))


def score():
    import numpy as np, pandas as pd
    import evaluate as ev, best_so_far
    from sklearn.metrics import balanced_accuracy_score as bas
    cat = lambda d, yrs: np.concatenate([d[y] for y in yrs])
    P = {n: ev.load_config(n)[0] for n in json.load(open(os.path.join(SHIPPED, "configs.json")))}
    P["BEST SO FAR (0.7 LGBM-63+A5 + 0.3 NN+D4)"] = {y: sum(w * P[n][y] for n, w in BEST) for y in range(2014, 2025)}
    P["70/30 LightGBM-63 + NN 256x256 (no demand)"] = {y: .7 * P["lgbm63"][y] + .3 * P["nn256x256"][y] for y in range(2014, 2025)}
    rows = {}
    for k, p in P.items():
        r = {"BA_2014 (tuning year)": 100 * bas(ev.Y[2014], p[2014] > .5)}
        for lab, yrs in (("BA_2015-24", ev.ALL), ("BA_2015-19", ev.SEL), ("BA_2020-24", ev.CONF)):
            r[lab] = 100 * bas(cat(ev.Y, yrs), cat(p, yrs) > .5)
        r["accuracy_2015-24"] = 100 * np.mean((cat(p, ev.ALL) > .5) == cat(ev.Y, ev.ALL))
        rows[k] = r
    T = pd.DataFrame(rows).T.sort_values("BA_2015-24", ascending=False)
    with pd.option_context("display.width", 200):
        print("\n== Walk-forward results, threshold 0.5 (package model = lgbm31; it uses 0.5 since commit 9651c2c)")
        print(T.round(3).to_string())
    T.round(4).to_csv(os.path.join(REPO, "results", "walkforward", "results_table.csv"))
    best_so_far.BEST["parts"] = BEST
    best_so_far.report({k: v for k, v in P.items() if not k.startswith("BEST")}, title="Every model vs the best so far")
    print("\n== BA by year")
    print(pd.DataFrame({k[:34]: {y: round(100 * bas(ev.Y[y], P[k][y] > .5), 2) for y in ev.ALL}
                        for k in ("BEST SO FAR (0.7 LGBM-63+A5 + 0.3 NN+D4)", "lgbm63_A5_blend", "nn256_D4_blend", "lgbm63", "lgbm31")}).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", nargs="*", default=None, help="configuration names to refit (default: none)")
    ap.add_argument("--fit-ends", nargs="*", type=int, default=None, help="limit refits to these fit_end years")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import harness
    if not os.path.exists(os.path.join(harness.E, "X.npy")):
        print("Building the data cache (rows up to 2024-12-31 only) ...", flush=True)
        harness.build_cache()
    if not os.path.exists(os.path.join(SHIPPED, "anubhav_blend_forecast.npy")):
        subprocess.run([sys.executable, os.path.join(HERE, "build_anubhav_blend.py")], check=True)
    if a.refit:
        refit(a.refit, a.fit_ends, a.workers)
    score()
