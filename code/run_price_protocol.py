"""Run the C2 price candidate through the three-stage protocol of run_experiments.py.

C2 (five LightGBM members, seeds 9417-9421, on the 571 price features plus 26
weather-shape inputs) joins Stage 1 as one more price candidate. It is evaluated
with the same rows, labels, simulation loop, drift detectors and metrics as the
original candidates in run_experiments.py:

  Stage 1  fit through 2013, choose the cutoff on 2014 (same grid and rule as
           select_models), compare with the original 2014 scores, refit
           through 2014 (the frozen model).
  Stage 2  frozen predictions 2015 - Oct 2026; if C2 is selected, the four
           adaptation strategies via run_experiments.simulate_policy.
  Stage 3  strategy with the best 2015-2024 balanced accuracy.

The fixed 0.50 cutoff is reported as a sensitivity check and never used to select.

  python run_experiments.py --output ../results/protocol/run_experiments
  python run_price_protocol.py
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from run_experiments import STRATEGIES, simulate_policy  # noqa: E402  (loads forecast_package's model)
from model import load_data, market_days, training_table  # noqa: E402
from analysis_tools import day_type_analysis, metric_values, period_metrics  # noqa: E402

sys.path.append(os.path.join(HERE, "price_v3"))
from recipe import fit_price, weather_shape_names  # noqa: E402  (the package's own C2 recipe)

# price_v2/model.py is also called "model"; load it under its own name.
_spec = importlib.util.spec_from_file_location(
    "price_v2_model", os.path.join(HERE, "price_v2", "model.py"))
price_v2_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(price_v2_model)

C2 = {"name": "price_c2", "seeds": [9417, 9418, 9419, 9420, 9421],
      "params": {"num_leaves": 63, "learning_rate": 0.02, "n_estimators": 1200,
                 "colsample_bytree": 0.5, "min_child_samples": 300}}
CUTOFF_GRID = np.linspace(0.2, 0.8, 61)  # as in train.select_models
CUTOFFS = ("tuned_2014", "fixed_0.50")
SIMULATION_START = pd.Timestamp("2015-01-01")


def c2_table(frame, index):
    """The package's 571 base columns, then the 26 weather-shape columns, on given rows."""
    shape = weather_shape_names()
    features = price_v2_model.feature_frame(frame, weather_shape=True)
    columns = [col for col in features.columns if col not in shape] + shape
    return features.loc[index, columns]


def fit_bag(task, config, x, y):
    """One recipe.fit_price member per seed; days with missing inputs are dropped, as in the package."""
    if task != "price":
        raise ValueError("C2 is a price model")
    complete = pd.Series(np.isfinite(x.to_numpy()).all(axis=1), index=x.index)
    complete = complete.groupby(market_days(x.index)).transform("all").to_numpy()
    x, y = x.loc[complete], y.loc[complete]
    members = [fit_price(x, y["price"], seed, config.get("params")) for seed in config["seeds"]]
    days = market_days(x.index)
    return {"task": task, "config": config, "features": list(x.columns), "members": members,
            "threshold": config.get("threshold", 0.5),
            "training_start": str(days.min()), "training_end": str(days.max())}


def bag_probability(bundle, x):
    x = x.loc[:, bundle["features"]]
    return np.mean([member.predict_proba(x)[:, 1] for member in bundle["members"]], axis=0)


def predict_bag(bundle, x):
    return (bag_probability(bundle, x) > bundle["threshold"]).astype(np.int64)


def cached_fit(folder):
    """fit_bag with an on-disk cache keyed by recipe and training rows, not by cutoff."""
    os.makedirs(folder, exist_ok=True)

    def fit(task, config, x, y):
        recipe = json.dumps({k: v for k, v in config.items() if k != "threshold"}, sort_keys=True)
        key = hashlib.sha256(f"{recipe}|{x.index[0]}|{x.index[-1]}|{len(x)}".encode()).hexdigest()
        path = os.path.join(folder, f"{config['name']}_{market_days(x.index[-1:])[0].date()}_{key[:16]}.joblib")
        if os.path.exists(path):
            bundle = joblib.load(path)
        else:
            bundle = fit_bag(task, config, x, y)
            joblib.dump(bundle, path + ".tmp")
            os.replace(path + ".tmp", path)
        return dict(bundle, threshold=config.get("threshold", 0.5))
    return fit


def stage1(x, y, config, fit=fit_bag):
    """Fit through 2013, score 2014 at the tuned and fixed cutoffs, refit through 2014."""
    days = market_days(x.index)
    train, valid = days.year <= 2013, days.year == 2014
    probability = bag_probability(fit("price", config, x.loc[train], y.loc[train]), x.loc[valid])
    actual = y.loc[valid, "price"].to_numpy()
    scores = [balanced_accuracy_score(actual, probability > cutoff) for cutoff in CUTOFF_GRID]
    result = {"cutoff": float(CUTOFF_GRID[int(np.argmax(scores))]), "scores": {}, "calibration": {}}
    for mode, cutoff in zip(CUTOFFS, (result["cutoff"], 0.5)):
        predicted = (probability > cutoff).astype(int)
        result["scores"][mode] = {
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted) * 100),
            "accuracy": float(np.mean(actual == predicted) * 100)}
        # Daily 2014 error: the error detector's reference, as in run_experiments.
        result["calibration"][mode] = (actual != predicted).reshape(-1, 48).mean(axis=1) * 100
    result["frozen"] = fit("price", config, x.loc[days.year <= 2014], y.loc[days.year <= 2014])
    return result


def run_policy(strategy, mode, stage, config, x, y, days, input_scores, input_reference,
               fit=fit_bag):
    cutoff = stage["cutoff"] if mode == "tuned_2014" else 0.5
    return simulate_policy("price", strategy, dict(stage["frozen"], threshold=cutoff),
                           dict(config, threshold=cutoff), x, y, days, input_scores,
                           input_reference, stage["calibration"][mode],
                           fit=fit, predict=predict_bag)


def save_tables(cache, x, labels):
    """Write the C2 table once so spawned workers memory-map it instead of rebuilding it."""
    os.makedirs(cache, exist_ok=True)
    np.save(os.path.join(cache, "c2_table.npy"), x.to_numpy(dtype=np.float32))
    labels.to_parquet(os.path.join(cache, "labels.parquet"))
    with open(os.path.join(cache, "c2_columns.json"), "w", encoding="utf-8") as handle:
        json.dump(list(x.columns), handle)


def _policy_job(job):
    """Runs one strategy at one cutoff, in a spawned worker (no inherited OpenMP state)."""
    strategy, mode, stage, config, cache, input_scores, input_reference = job
    path = os.path.join(cache, "policies", f"{strategy}_{mode}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    labels = pd.read_parquet(os.path.join(cache, "labels.parquet"))
    with open(os.path.join(cache, "c2_columns.json"), encoding="utf-8") as handle:
        columns = json.load(handle)
    x = pd.DataFrame(np.load(os.path.join(cache, "c2_table.npy"), mmap_mode="r"),
                     index=labels.index, columns=columns, copy=False)
    all_days = market_days(labels.index)
    days = all_days[all_days >= SIMULATION_START].unique()
    with threadpool_limits(limits=2):
        output = run_policy(strategy, mode, stage, config, x, labels, days, input_scores,
                            input_reference, fit=cached_fit(os.path.join(cache, "fits")))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(output, path)
    print(f"Policy {strategy} at {mode}: {len(output[1])} retrains", flush=True)
    return output


def run_jobs(jobs, workers):
    if workers <= 1:
        return [_policy_job(job) for job in jobs]
    results = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for mode in CUTOFFS:  # tuned fits are cached before the 0.50 runs reuse them
            results += list(pool.map(_policy_job, [job for job in jobs if job[1] == mode]))
    return results


def original_rows(path):
    with open(path, encoding="utf-8") as handle:
        validation = json.load(handle)["validation"]
    return [{"model": row["model"], "source": "run_experiments",
             "cutoff_tuned": row["config"]["threshold"], "BA_tuned": row["balanced_accuracy"],
             "accuracy_tuned": row["accuracy"], "BA_fixed_0.50": np.nan,
             "accuracy_fixed_0.50": np.nan}
            for row in validation if row["task"] == "price"]


def save_monthly_figures(folder, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    monthly = metrics[metrics["frequency"] == "M"]
    for (name, mode), subset in monthly.groupby(["model", "cutoff"], sort=False):
        dates = pd.to_datetime(subset["period"] + "-01")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(dates, subset["accuracy"], color="black", linewidth=1, label="accuracy")
        for interval, color in (("iid", "tab:orange"), ("day_boot", "tab:blue")):
            ax.fill_between(dates, subset[f"accuracy_{interval}_low"],
                            subset[f"accuracy_{interval}_high"], alpha=0.25, color=color,
                            label=f"95% {interval}")
        ax.set(xlabel="Month", ylabel="Price accuracy (%)", title=f"{name} ({mode})")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"monthly_{name}_{mode}.png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(ROOT, "data",
                        "nem_weather_halfhourly_2010_2026.csv"))
    parser.add_argument("--rerun", default=os.path.join(ROOT, "results", "protocol", "run_experiments"),
                        help="run_experiments.py output whose selection.json holds the original 2014 scores")
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "protocol", "price_c2"))
    parser.add_argument("--quick", action="store_true",
                        help="40 trees and 2 seeds: checks the pipeline, not a result")
    parser.add_argument("--resume", action="store_true", help="Reuse fits and policies cached in --output")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--selection-end", default="2024-12-31",
                        help="Choose the strategy on earlier simulation days; later days are holdout")
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error("--bootstrap must be at least 1000")
    cache = os.path.join(args.output, "cache")
    if os.path.exists(cache) and not args.resume:
        parser.error(f"{cache} exists: pass --resume or choose a new --output")
    os.makedirs(os.path.join(cache, "policies"), exist_ok=True)
    started = time.monotonic()
    config = dict(C2)
    if args.quick:
        config = dict(C2, name="price_c2_quick", seeds=C2["seeds"][:2],
                      params=dict(C2["params"], n_estimators=40))

    frame = load_data(args.data)
    features, labels = training_table(frame)
    x = c2_table(frame, features.index)
    stage = stage1(x, labels, config, cached_fit(os.path.join(cache, "fits")))
    rows = original_rows(os.path.join(args.rerun, "selection.json"))
    rows.append({"model": config["name"], "source": "run_price_protocol",
                 "cutoff_tuned": stage["cutoff"],
                 "BA_tuned": stage["scores"]["tuned_2014"]["balanced_accuracy"],
                 "accuracy_tuned": stage["scores"]["tuned_2014"]["accuracy"],
                 "BA_fixed_0.50": stage["scores"]["fixed_0.50"]["balanced_accuracy"],
                 "accuracy_fixed_0.50": stage["scores"]["fixed_0.50"]["accuracy"]})
    candidates = pd.DataFrame(rows)
    best = candidates.loc[candidates["BA_tuned"].idxmax()]  # first maximum: ties keep the original
    c2_selected = best["model"] == config["name"]
    simulate = c2_selected or args.quick  # --quick always exercises Stages 2-3 as a pipeline check
    os.makedirs(args.output, exist_ok=True)
    candidates.to_csv(os.path.join(args.output, "stage1.csv"), index=False)
    with open(os.path.join(args.output, "selection.json"), "w", encoding="utf-8") as handle:
        json.dump({"selected": {"model": best["model"], "source": best["source"],
                                "cutoff": float(best["cutoff_tuned"])},
                   "c2": {"config": config, "cutoff": stage["cutoff"], "scores": stage["scores"]},
                   "candidates": candidates.to_dict(orient="records")}, handle, indent=2)
    print(candidates.to_string(index=False), flush=True)
    print(f"Stage 1 selects {best['model']} ({best['source']})", flush=True)

    all_days = market_days(features.index)
    simulation = all_days >= SIMULATION_START
    days = all_days[simulation].unique()
    index = features.index[simulation]
    truth = {task: labels.loc[simulation, task].to_numpy().reshape(-1, 48)
             for task in ("demand", "price")}
    probability = bag_probability(stage["frozen"], x.loc[simulation])
    if simulate:
        save_tables(cache, x, labels)
    del x, features
    predictions = {("frozen_c2", mode): (probability > cutoff).astype(np.int64).reshape(-1, 48)
                   for mode, cutoff in zip(CUTOFFS, (stage["cutoff"], 0.5))}
    retrains, signals = [], []
    if simulate:
        types, _, _, _ = day_type_analysis(frame)
        reference = types.loc[types.index.year == 2014, "input_drift_score"].to_numpy()
        jobs = [(strategy, mode, stage, config, cache, types["input_drift_score"], reference)
                for mode in CUTOFFS for strategy in STRATEGIES]
        for job, (forecast, updates, log) in zip(jobs, run_jobs(jobs, args.workers)):
            strategy, mode = job[0], job[1]
            predictions[(f"adapted_c2_{strategy}", mode)] = forecast
            retrains += [dict(row, model="c2", cutoff=mode) for row in updates]
            signals += [dict(row, model="c2", cutoff=mode) for row in log]
        budget = pd.DataFrame(retrains)
        if len(budget) and budget.groupby(["strategy", "cutoff"]).size().max() > 6:
            raise AssertionError("Retraining budget exceeded")

    selection_mask = days <= pd.Timestamp(args.selection_end)
    if not selection_mask.any() or selection_mask.all():
        raise ValueError("Strategy selection requires nonempty selection and holdout periods")
    periods = {"simulation": np.ones(len(days), dtype=bool),
               "strategy_selection": selection_mask, "holdout": ~selection_mask}
    tables, totals = [], []
    for (name, mode), forecast in predictions.items():
        table = period_metrics(days, truth, {"demand": truth["demand"], "price": forecast},
                               bootstrap=args.bootstrap)
        table = table.drop(columns=[col for col in table if col.startswith(("MAPE", "MAE"))])
        table.insert(0, "cutoff", mode)
        table.insert(0, "model", name)
        tables.append(table)
        for period, mask in periods.items():
            values = metric_values(truth["demand"][mask].ravel(), truth["price"][mask].ravel(),
                                   truth["demand"][mask].ravel(), forecast[mask].ravel())
            totals.append({"model": name, "cutoff": mode, "period": period,
                           "accuracy": values["accuracy"],
                           "balanced_accuracy": values["balanced_accuracy"]})
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(os.path.join(args.output, "period_metrics.csv"), index=False)
    aggregate = pd.DataFrame(totals)
    aggregate.to_csv(os.path.join(args.output, "aggregate_metrics.csv"), index=False)
    pd.DataFrame(retrains).to_csv(os.path.join(args.output, "retrains.csv"), index=False)
    pd.DataFrame(signals).to_csv(os.path.join(args.output, "detector_signals.csv"), index=False)
    data = {"price_up": truth["price"].ravel()}
    data.update({f"{name}_{mode}": forecast.ravel() for (name, mode), forecast in predictions.items()})
    pd.DataFrame(data, index=index).to_parquet(os.path.join(args.output, "half_hour_predictions.parquet"))
    save_monthly_figures(os.path.join(args.output, "figures"), metrics)

    if simulate:
        chosen = aggregate[(aggregate["period"] == "strategy_selection")
                           & (aggregate["cutoff"] == "tuned_2014")
                           & aggregate["model"].str.startswith("adapted_c2_")]
        strategy = chosen.loc[chosen["balanced_accuracy"].idxmax(), "model"].removeprefix("adapted_c2_")
        final = {"price": {"name": config["name"], "selected_by_protocol": bool(c2_selected),
                           "config": dict(config, threshold=stage["cutoff"]),
                           **STRATEGIES[strategy], "strategy": strategy,
                           "training_end": "2014-12-31" if strategy == "frozen" else None}}
    else:
        final = {"price": {"name": best["model"], "source": "run_experiments",
                           "final_config": os.path.join(args.rerun, "final_config.json")}}
    final.update(strategy_selection_end=args.selection_end, quick=args.quick)
    with open(os.path.join(args.output, "final_config.json"), "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)
    with open(os.path.join(args.output, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"seconds": time.monotonic() - started, "data": args.data,
                   "data_end": str(days.max()), "rerun": args.rerun, "config": config,
                   "c2_selected": bool(c2_selected), "cutoff_grid": [0.2, 0.8, 61],
                   "simulation_start": str(SIMULATION_START.date()),
                   "strategy_selection_end": args.selection_end, "bootstrap": args.bootstrap,
                   "workers": args.workers, "threads": 2,
                   "reference": "Detectors calibrated on 2014; PCA/clustering fit on 2010-2013"},
                  handle, indent=2)
    print(aggregate[aggregate["period"] != "simulation"].to_string(index=False), flush=True)
    print(f"Final price policy: {json.dumps(final['price'], default=str)}", flush=True)
    print(f"Outputs saved to {args.output} in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
