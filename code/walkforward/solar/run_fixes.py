"""Pure test (not in the repo): two fixes for the fading demand-feature gain. Walk-forward only: the harness data ends
2024-12-31, so nothing here can read or predict 2025-26.

stage1: demand streams  ctrl  = Anubhav's recipe (0.5 LightGBM + 0.5 ten-epoch MLP) rebuilt by this code
                        solar = the same + Suhan's 26 weather-shape inputs + 6 solar-sensitivity inputs (fix b)
        demand-free twins for fix (a): LGBM-63+ws from 2011 and Suhan C2+ws from 2011, 5 seeds each
stage2: (only if fix b passes its demand gate) the champion's demand members refitted on the solar stream
Folds of the demand stream = the schedule of runs/anubhav_blend_forecast.npy: yearly to 2015, then every second year."""
import json, os, subprocess, sys, tempfile, time, shutil
os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2", MKL_NUM_THREADS="2")
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
S = os.path.dirname(os.path.abspath(__file__)); E = os.path.dirname(S); sys.path.insert(0, E)
import numpy as np, pandas as pd
import harness

FOLDS = [(2010, [2011]), (2011, [2012]), (2012, [2013]), (2013, [2014]), (2014, [2015]), (2015, [2016, 2017]),
         (2017, [2018, 2019]), (2019, [2020, 2021]), (2021, [2022, 2023]), (2023, [2024])]
SEEDS = (9417, 9418, 9419, 9420, 9421)
LGBM63 = {"leaves": 63, "trees": 900, "learning_rate": 0.025}
C2 = {"config": {"leaves": 63, "trees": 1200, "learning_rate": 0.02}, "lgb_params": {"colsample_bytree": 0.5, "min_child_samples": 300}}
TWINS = {f"lgbm63ws_from2011_s{s}": {"kind": "lgbx", "config": LGBM63, "seed": s, "weather_shape": True, "start_year": 2011}
         for s in SEEDS}
TWINS.update({f"suhan_c2ws_from2011_s{s}": dict({"kind": "lgbx", "seed": s, "weather_shape": True, "start_year": 2011}, **C2)
              for s in SEEDS})
STAGE2 = {f"lgbm63_D4ws_solar_s{s}": {"kind": "lgbx", "config": LGBM63, "seed": s, "weather_shape": True, "start_year": 2011,
                                      "demand": "D4", "demand_source": "solar_blend"} for s in SEEDS}
STAGE2.update({f"suhan_c2ws_A5_solar_s{s}": dict({"kind": "lgbx", "seed": s, "weather_shape": True, "start_year": 2011,
                                                  "demand": "A5", "demand_source": "solar_blend"}, **C2) for s in SEEDS})
STAGE2["nn256_D4_solar"] = {"kind": "sk", "config": {"family": "neural_net", "hidden": [256, 256], "dropout": 0.5,
                            "learning_rate": 0.0003, "weight_decay": 0.0001, "batch_size": 1024, "epochs": 10,
                            "seeds": [0, 1, 2, 3, 4], "log": "price", "winsorise": None, "bootstrap": False},
                            "start_year": 2011, "demand": "D4", "demand_source": "solar_blend"}
WS = json.load(open(f"{harness.RUNS}/weather_shape_names.json"))
SENS = json.load(open(f"{harness.RUNS}/solar_sens_names.json"))
MLP_COLS = json.load(open(f"{harness.REPO}/code/demand_feature_columns.json"))


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    open(f"{S}/progress.log", "a").write(line + "\n")


def demand_inputs(variant):
    X, truth, days = harness.load()
    if variant == "solar":
        X = X.copy()
        for k, n in enumerate(WS):
            X[n] = np.load(f"{harness.RUNS}/weather_shape.npy", mmap_mode="r")[:, k]
        for k, n in enumerate(SENS):
            X[n] = np.load(f"{harness.RUNS}/solar_sens.npy", mmap_mode="r")[:, k]
    return X, truth, days


def demand_job(part, variant, fit_end, years):
    out = f"{harness.RUNS}/dstream_{variant}_{part}__{fit_end}.npy"
    if os.path.exists(out):
        return out
    X, truth, days = demand_inputs(variant)
    tr, pr = np.asarray(days.year <= fit_end), np.asarray(days.year.isin(years))
    t0 = time.time()
    if part == "lgbm":
        from train import fit_model
        from model import predict_task
        model = fit_model("demand", dict(harness.DEMAND_CFG, n_jobs=2), X[tr], truth[tr])
        pred, cols = predict_task(model, X[pr]), list(model["features"])
    else:
        cols = MLP_COLS + (WS + SENS if variant == "solar" else [])
        tmp = tempfile.mkdtemp(dir=S)
        pd.to_pickle((X.loc[tr, cols], truth.loc[tr, "demand"], X.loc[pr, cols]), f"{tmp}/in.pkl")
        subprocess.run([sys.executable, f"{S}/mlp_fold.py", f"{tmp}/in.pkl", f"{tmp}/out.npy"], check=True,
                       env=dict(os.environ), stdout=subprocess.DEVNULL)
        pred = np.load(f"{tmp}/out.npy")
        shutil.rmtree(tmp)
    np.save(out, np.asarray(pred, np.float64))
    json.dump({"seconds": time.time() - t0, "n_features": len(cols), "train_rows": int(tr.sum()),
               "fit_end": fit_end, "years": years}, open(out + ".json", "w"))
    return out


def price_job(name, spec, fit_end, years):
    harness.fit_job(name, spec, fit_end, years)
    return name


def assemble(variant):
    X, truth, days = harness.load()
    f = np.full(len(days), np.nan)
    for fit_end, years in FOLDS:
        m = np.asarray(days.year.isin(years))
        a = np.load(f"{harness.RUNS}/dstream_{variant}_lgbm__{fit_end}.npy")
        b = np.load(f"{harness.RUNS}/dstream_{variant}_mlp__{fit_end}.npy")
        f[m] = 0.5 * (a + b)
    np.save(f"{harness.RUNS}/{variant}_blend_forecast.npy", f)
    return f


def run(jobs, workers):
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        futs = {pool.submit(fn, *args): args for fn, args in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                log(f"done {k}/{len(jobs)} {futs[fut]}")
            except Exception as exc:
                log(f"FAILED {futs[fut]}: {exc!r}")


if __name__ == "__main__":
    stage, workers = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if stage == "stage1":
        jobs = [(demand_job, (part, v, fe, ys)) for fe, ys in reversed(FOLDS) for v in ("solar", "ctrl") for part in ("lgbm", "mlp")]
        twins = [(price_job, (n, spec, fe, ys)) for fe, ys in reversed(harness.SCHEDULE) for n, spec in TWINS.items()
                 if not os.path.exists(f"{harness.RUNS}/{n}__{fe}.npz")]
        log(f"stage1: {len(jobs)} demand jobs, {len(twins)} twin fits, {workers} workers")
        run(jobs + twins, workers)
        for v in ("ctrl", "solar"):
            assemble(v)
        log("stage1 finished; streams assembled")
    elif stage == "assemble":
        for v in ("ctrl", "solar"):
            assemble(v)
    elif stage == "stage2":
        jobs = [(price_job, (n, spec, fe, ys)) for fe, ys in reversed(harness.SCHEDULE) for n, spec in STAGE2.items()
                if not os.path.exists(f"{harness.RUNS}/{n}__{fe}.npz")]
        log(f"stage2: {len(jobs)} fits, {workers} workers")
        run(jobs, workers)
        log("stage2 finished")
