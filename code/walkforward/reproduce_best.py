"""Reproduce the best price-direction configuration so far on the team walk-forward (2015-2024, threshold 0.5).

CHAMPION (2026-11-19):
    P(up) = 0.7 * mean over 10 LightGBM members + 0.3 * mean over 2 neural nets
    LightGBM members: 5 x LightGBM-63 + D4 demand features + 26 weather-shape inputs (seeds 9417-9421, our settings)
                      5 x Suhan's recipe (63 leaves, 1200 trees, lr 0.02, colsample 0.5, min_child 300)
                          + 26 weather-shape inputs + Anubhav's 5 demand features (A5)   (seeds 9417-9421)
    Neural nets:      NN 256x256 + D4 demand features;  NN 64 (no demand features)
    All demand features use Anubhav's day-ahead demand forecast (0.5 LightGBM + 0.5 MLP); demand models train from 2011.
PREVIOUS BEST (commit feb079b): 0.7 * LightGBM-63+A5 + 0.3 * NN 256x256+D4.

Usage (from the repo root, with the project venv):
    python code/walkforward/reproduce_best.py                  # score the shipped predictions (a few minutes)
    python code/walkforward/reproduce_best.py --refit lgbm63_D4ws_s9417 nn64 --fit-ends 2013
        # refit those configurations for the listed walk-forward fits into results/walkforward/refit/ and compare

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
S5 = (9417, 9418, 9419, 9420, 9421)
CHAMPION = ([(f"lgbm63_D4ws_s{s}", 0.07) for s in S5] + [(f"suhan_c2ws_A5_s{s}", 0.07) for s in S5]
            + [("nn256_D4_blend", 0.15), ("nn64", 0.15)])
PREVIOUS = [("lgbm63_A5_blend", 0.7), ("nn256_D4_blend", 0.3)]


def _fit(args):
    name, spec, fit_end, years, runs = args
    os.environ["WALKFORWARD_RUNS"] = runs
    import harness
    harness.RUNS = runs
    return harness.fit_job(name, spec, fit_end, years)


def derived_inputs(runs):
    """Inputs rebuilt locally (not shipped): continuous targets, weather-shape and negative-price arrays."""
    env = dict(os.environ, WALKFORWARD_RUNS=runs, PYTHONPATH=os.pathsep.join(sys.path[:2]))
    if not os.path.exists(os.path.join(runs, "targets.npz")):
        subprocess.run([sys.executable, os.path.join(HERE, "build_targets.py")], check=True, env=env)
    if not os.path.exists(os.path.join(runs, "weather_shape.npy")):
        subprocess.run([sys.executable, os.path.join(HERE, "build_weather_shape.py")], check=True, env=env)
    if not os.path.exists(os.path.join(runs, "negprice.npy")):
        subprocess.run([sys.executable, os.path.join(HERE, "build_negative_features.py")], check=True, env=env)


def refit(names, fit_ends, workers):
    import numpy as np
    import harness
    runs = os.path.join(REPO, "results", "walkforward", "refit")
    os.makedirs(runs, exist_ok=True)
    for f in ("anubhav_blend_forecast.npy", "weather_shape_names.json"):
        shutil.copy(os.path.join(SHIPPED, f), runs)
    derived_inputs(runs)
    specs = json.load(open(os.path.join(SHIPPED, "configs.json")))
    jobs = [(n, specs[n], fe, ys, runs) for n in names for fe, ys in harness.SCHEDULE if not fit_ends or fe in fit_ends]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_fit, jobs))
    print("\nRefit vs shipped predictions (max |probability difference|):")
    for n, _, fe, ys, _ in jobs:
        a, b = np.load(f"{runs}/{n}__{fe}.npz"), np.load(f"{SHIPPED}/{n}__{fe}.npz")
        print(f"  {n:22s} fit<={fe}: " + ", ".join(f"{y}: {float(np.max(np.abs(a[f'y{y}'] - b[f'y{y}']))):.2e}" for y in ys))


def score():
    import numpy as np, pandas as pd
    import evaluate as ev, best_so_far
    from sklearn.metrics import balanced_accuracy_score as bas
    cat = lambda d, yrs: np.concatenate([d[y] for y in yrs])
    YRS = range(2014, 2025)
    P = {n: ev.load_config(n)[0] for n in json.load(open(os.path.join(SHIPPED, "configs.json")))}
    comb = lambda parts: {y: sum(w * P[n][y] for n, w in parts) for y in YRS}
    mean = lambda names: {y: np.mean([P[n][y] for n in names], axis=0) for y in YRS}
    R = {"CHAMPION (0.7 x 10 LightGBM + 0.3 x 2 NN)": comb(CHAMPION),
         "previous best (0.7 LGBM-63+A5 + 0.3 NN+D4)": comb(PREVIOUS),
         "LightGBM part: LGBM-63+D4+ws, 5 seeds": mean([f"lgbm63_D4ws_s{s}" for s in S5]),
         "LightGBM part: Suhan recipe + A5, 5 seeds": mean([f"suhan_c2ws_A5_s{s}" for s in S5]),
         "NN part: NN 256x256 + D4": P["nn256_D4_blend"], "NN part: NN 64": P["nn64"],
         "LightGBM-63 (no demand)": P["lgbm63"], "package LightGBM-31 at 0.5": P["lgbm31"],
         "package LightGBM-31 at 0.57 (reported 83.163 setup)": {y: P["lgbm31"][y] - 0.07 for y in YRS}}
    rows = {}
    for k, p in R.items():
        r = {"BA_2014 (tuning year)": 100 * bas(ev.Y[2014], p[2014] > .5)}
        for lab, yrs in (("BA_2015-24", ev.ALL), ("BA_2015-19", ev.SEL), ("BA_2020-24", ev.CONF)):
            r[lab] = 100 * bas(cat(ev.Y, yrs), cat(p, yrs) > .5)
        r["accuracy_2015-24"] = 100 * np.mean((cat(p, ev.ALL) > .5) == cat(ev.Y, ev.ALL))
        rows[k] = r
    T = pd.DataFrame(rows).T.sort_values("BA_2015-24", ascending=False)
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print("\n== Walk-forward results, threshold 0.5"); print(T.round(3).to_string())
    T.round(4).to_csv(os.path.join(REPO, "results", "walkforward", "results_table.csv"))
    best_so_far.BEST = {"label": "CHAMPION", "parts": CHAMPION}
    best_so_far.report({k: v for k, v in R.items() if not k.startswith("CHAMPION")}, title="Every model vs the CHAMPION")
    print("\n== BA by year")
    show = ["CHAMPION (0.7 x 10 LightGBM + 0.3 x 2 NN)", "previous best (0.7 LGBM-63+A5 + 0.3 NN+D4)",
            "LightGBM part: LGBM-63+D4+ws, 5 seeds", "LightGBM part: Suhan recipe + A5, 5 seeds", "package LightGBM-31 at 0.5"]
    print(pd.DataFrame({k[:30]: {y: round(100 * bas(ev.Y[y], R[k][y] > .5), 2) for y in ev.ALL} for k in show}).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", nargs="*", default=None, help="configuration names to refit (default: none)")
    ap.add_argument("--fit-ends", nargs="*", type=int, default=None, help="limit refits to these fit_end years")
    ap.add_argument("--workers", type=int, default=1)
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
