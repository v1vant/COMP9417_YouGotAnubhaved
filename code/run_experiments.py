"""Run all three-stage project experiments without temporal information leakage."""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
from model import load_data, market_days, predict_task, training_table
from train import fit_model, select_models
from analysis_tools import (
    OnlineDetector, day_type_analysis, metric_values, period_metrics, save_figures,
)

STRATEGIES = {
    "frozen": {"window_years": None},
    "periodic_expanding": {"window_years": None},
    "periodic_sliding": {"window_years": 2},
    "drift_sliding": {"window_years": 4},
}


def simulate_policy(task, strategy, initial, config, features, truth, simulation_days,
                    input_scores, input_reference, error_reference):
    """Predict D first, then observe D; any triggered replacement starts on D+1."""
    output = np.empty((len(simulation_days), 48), dtype=float if task == "demand" else int)
    model = initial
    input_detector = OnlineDetector(input_reference)
    error_detector = OnlineDetector(error_reference)
    feature_days = market_days(features.index)
    positions = np.flatnonzero(feature_days >= simulation_days[0])
    if len(positions) != len(simulation_days) * 48:
        raise ValueError("Simulation must consist of consecutive complete days to data end")
    last_fit = pd.Timestamp("2014-12-31")
    retrains, signals = [], []
    pending = False
    cached, cache_end = None, 0
    for i, day in enumerate(simulation_days):
        scheduled = strategy.startswith("periodic") and day.month == 1 and day.day == 1 \
            and day.year in (2016, 2018, 2020, 2022, 2024, 2026)
        if (pending or scheduled) and len(retrains) < 6:
            mask = feature_days < day
            window = STRATEGIES[strategy]["window_years"]
            if window:
                mask &= feature_days >= day - pd.DateOffset(years=window)
            model = fit_model(task, config, features.loc[mask], truth.loc[mask])
            last_fit = day - pd.Timedelta(days=1)
            retrains.append({
                "task": task, "strategy": strategy, "effective_day": str(day.date()),
                "training_end": str(feature_days[mask].max().date()),
                "training_start": str(feature_days[mask].min().date()),
                "training_rows": int(mask.sum()), "cause": "drift" if pending else "schedule",
            })
            cached = None
            print(f"Retrain {task}/{strategy} effective {day.date()} ({len(retrains)}/6)",
                  flush=True)
        pending = False
        if cached is None or i >= cache_end:
            cache_end = min(i + 64, len(simulation_days))
            rows = features.iloc[positions[i * 48]:positions[(cache_end - 1) * 48 + 47] + 1]
            cached = predict_task(model, rows).reshape(-1, 48)
            cache_start = i
        output[i] = cached[i - cache_start]
        actual = truth.iloc[positions[i * 48]:positions[i * 48 + 47] + 1][task].to_numpy()
        error = (float(np.mean(np.abs(actual - output[i]) / actual) * 100)
                 if task == "demand" else float(np.mean(actual != output[i]) * 100))
        input_signal = input_detector.observe(day, input_scores.loc[day])
        error_signal = error_detector.observe(day, error)
        signals.append({
            "day": str(day.date()), "task": task, "strategy": strategy,
            "input_statistic": input_detector.statistic,
            "input_threshold": input_detector.threshold, "input_signal": input_signal,
            "error_statistic": error_detector.statistic,
            "error_threshold": error_detector.threshold, "error_signal": error_signal,
            "observed_error": error,
        })
        if strategy == "drift_sliding" and (input_signal or error_signal) \
                and len(retrains) < 6 and (not retrains or (day - last_fit).days >= 180):
            pending = True
    return output, retrains, signals


def baseline_predictions(frame, index):
    day_means = frame["nsw_price"].groupby(market_days(frame.index)).mean().shift(1)
    threshold = pd.Series(market_days(index), index=index).map(day_means)
    result = {}
    for lag in (1, 7):
        result[f"baseline_{lag}d"] = {
            "demand": frame["nsw_demand"].shift(48 * lag).loc[index].to_numpy().reshape(-1, 48),
            "price": (frame["nsw_price"].shift(48 * lag).loc[index] > threshold)
                     .to_numpy(dtype=int).reshape(-1, 48),
        }
    return result


def validation_analysis(frame, features, labels, selected, frozen, output):
    days = market_days(features.index)
    train_mask = days.year <= 2013
    valid_mask = days.year == 2014
    index = features.index[valid_mask]
    baselines = baseline_predictions(frame, index)
    baseline_rows = []
    for name, forecast in baselines.items():
        baseline_rows.append({
            "model": name,
            **metric_values(labels.loc[valid_mask, "demand"], labels.loc[valid_mask, "price"],
                            forecast["demand"].ravel(), forecast["price"].ravel()),
        })
    pd.DataFrame(baseline_rows).to_csv(os.path.join(output, "validation_baselines.csv"), index=False)
    ablations = []
    for task in ("demand", "price"):
        for name, columns in (
            ("without_weather", [col for col in features
                                 if not col.startswith(("syd_", "mel_", "bne_"))]),
            ("NSW_only", [col for col in features
                          if not col.startswith(("vic_", "qld_", "mel_", "bne_"))]),
        ):
            model = fit_model(task, selected[task]["config"],
                              features.loc[train_mask, columns], labels.loc[train_mask])
            prediction = predict_task(model, features.loc[valid_mask])
            actual = labels.loc[valid_mask, task].to_numpy()
            row = {"task": task, "ablation": name}
            if task == "demand":
                row.update(MAPE=float(np.mean(np.abs(actual - prediction) / actual) * 100),
                           MAE=float(np.mean(np.abs(actual - prediction))))
            else:
                from sklearn.metrics import balanced_accuracy_score
                row.update(accuracy=float(np.mean(actual == prediction) * 100),
                           balanced_accuracy=float(balanced_accuracy_score(actual, prediction) * 100))
            ablations.append(row)
    pd.DataFrame(ablations).to_csv(os.path.join(output, "validation_ablations.csv"), index=False)
    importance = []
    for name, bundle in frozen.items():
        estimator = bundle["estimator"]
        if hasattr(estimator, "booster_"):
            values = estimator.booster_.feature_importance(importance_type="gain")
            kind = "training_split_gain"
        else:
            values = np.abs(estimator.steps[-1][1].coef_).ravel()
            kind = "absolute_standardized_coefficient"
        importance.extend({"model": name, "feature": column, "importance": float(value),
                           "method": kind} for column, value in zip(bundle["features"], values))
    pd.DataFrame(importance).to_csv(os.path.join(output, "feature_importance.csv"), index=False)
    frame.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_csv(
        os.path.join(output, "data_summary.csv"))
    class_balance = pd.DataFrame({
        "year": days.year, "price_up": labels["price"].to_numpy(),
        "previous_mean_price": features["price_threshold"].to_numpy(),
    }).groupby("year").agg(
        class_1_share=("price_up", "mean"), rows=("price_up", "size"),
        previous_mean_price=("previous_mean_price", "mean"))
    class_balance.to_csv(os.path.join(output, "class_balance_by_year.csv"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(ROOT, "data",
                        "nem_weather_halfhourly_2010_2026.csv"))
    parser.add_argument("--output", default=os.path.join(ROOT, "results"))
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer trees; still run the complete chronological simulation")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--selection-end", default="2024-12-31",
                        help="Choose adaptation on earlier simulation; later days are holdout")
    parser.add_argument("--resume", action="store_true", help="Reuse completed selection/policies")
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error("--bootstrap must be at least 1000")
    started = time.monotonic()
    os.makedirs(args.output, exist_ok=True)
    cache_dir = os.path.join(args.output, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    frame = load_data(args.data)
    features, labels = training_table(frame)
    all_days = market_days(features.index)
    simulation_mask = all_days >= pd.Timestamp("2015-01-01")
    simulation_index = features.index[simulation_mask]
    days = all_days[simulation_mask].unique()
    truth = {task: labels.loc[simulation_mask, task].to_numpy().reshape(-1, 48)
             for task in ("demand", "price")}
    fingerprint = {
        "rows": len(frame), "end": str(frame.index[-1]), "quick": args.quick,
        "feature_columns": list(features.columns),
        "source_hashes": {},
    }
    import hashlib
    for source in ("run_experiments.py", "analysis_tools.py",
                   os.path.join("forecast_package", "model.py"),
                   os.path.join("forecast_package", "train.py")):
        with open(os.path.join(HERE, source), "rb") as handle:
            fingerprint["source_hashes"][source] = hashlib.sha256(handle.read()).hexdigest()
    # Cache validity also depends on values, not only shape and source code.
    fingerprint["data_hash"] = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()
    selection_path = os.path.join(cache_dir, "stage1.joblib")
    cached_selection = joblib.load(selection_path) if args.resume and os.path.exists(selection_path) else None
    if cached_selection is not None and cached_selection["fingerprint"] == fingerprint:
        selected, validation, frozen, calibration = cached_selection["payload"]
    else:
        selected, validation, frozen, calibration = select_models(features, labels, args.quick)
        joblib.dump({"fingerprint": fingerprint,
                     "payload": (selected, validation, frozen, calibration)}, selection_path)
    with open(os.path.join(args.output, "selection.json"), "w", encoding="utf-8") as handle:
        json.dump({"selected": selected, "validation": validation}, handle, indent=2)
    validation_analysis(frame, features, labels, selected, frozen, args.output)
    types, shares, typical, clustering = day_type_analysis(frame)
    types.to_csv(os.path.join(args.output, "day_types.csv"))
    shares.to_csv(os.path.join(args.output, "day_type_shares.csv"))
    joblib.dump(clustering, os.path.join(args.output, "day_type_models.joblib"))
    with open(os.path.join(args.output, "day_type_selection.json"), "w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in clustering.items()
                   if key not in ("pca", "clustering")}, handle, indent=2)
    reference = types.loc[types.index.year == 2014, "input_drift_score"].to_numpy()
    predictions = baseline_predictions(frame, simulation_index)
    for suffix in sorted({name.split("_", 1)[1] for name in frozen}):
        predictions[f"frozen_{suffix}"] = {
            task: predict_task(frozen[f"{task}_{suffix}"], features.loc[simulation_mask])
                  .reshape(-1, 48) for task in ("demand", "price")
        }
    retrains, signals = [], []
    for strategy in STRATEGIES:
        forecasts = {}
        for task in ("demand", "price"):
            policy_path = os.path.join(cache_dir, f"{task}_{strategy}.joblib")
            cached_policy = joblib.load(policy_path) if args.resume and os.path.exists(policy_path) else None
            if cached_policy is not None and cached_policy["fingerprint"] == fingerprint:
                forecast, updates, detector_log = cached_policy["payload"]
            else:
                forecast, updates, detector_log = simulate_policy(
                    task, strategy, frozen[selected[task]["name"]], selected[task]["config"],
                    features, labels, days, types["input_drift_score"], reference,
                    calibration[task].reshape(-1, 48).mean(axis=1))
                joblib.dump({"fingerprint": fingerprint,
                             "payload": (forecast, updates, detector_log)}, policy_path)
            forecasts[task] = forecast
            retrains.extend(updates)
            signals.extend(detector_log)
        predictions[f"adapted_{strategy}"] = forecasts
    retrain_frame = pd.DataFrame(retrains)
    if len(retrain_frame) and retrain_frame.groupby(["task", "strategy"]).size().max() > 6:
        raise AssertionError("Retraining budget exceeded")
    retrain_frame.to_csv(os.path.join(args.output, "retrains.csv"), index=False)
    pd.DataFrame(signals).to_csv(os.path.join(args.output, "detector_signals.csv"), index=False)
    metric_tables, totals = [], []
    selection_mask = days <= pd.Timestamp(args.selection_end)
    holdout_mask = days > pd.Timestamp(args.selection_end)
    if not selection_mask.any() or not holdout_mask.any():
        raise ValueError("Strategy selection requires nonempty selection and holdout periods")
    for name, forecast in predictions.items():
        table = period_metrics(days, truth, forecast, bootstrap=args.bootstrap)
        table.insert(0, "model", name)
        metric_tables.append(table)
        for period, mask in (("simulation", np.ones(len(days), dtype=bool)),
                             ("strategy_selection", selection_mask),
                             ("holdout", holdout_mask)):
            totals.append({
                "model": name, "period": period,
                **metric_values(truth["demand"][mask].ravel(), truth["price"][mask].ravel(),
                                forecast["demand"][mask].ravel(), forecast["price"][mask].ravel()),
            })
    metrics = pd.concat(metric_tables, ignore_index=True)
    metrics.to_csv(os.path.join(args.output, "period_metrics.csv"), index=False)
    aggregate = pd.DataFrame(totals)
    aggregate.to_csv(os.path.join(args.output, "aggregate_metrics.csv"), index=False)
    print(aggregate[aggregate["period"] == "holdout"].to_string(index=False), flush=True)
    data = {"nsw_demand": truth["demand"].ravel(), "price_up": truth["price"].ravel()}
    for name, forecast in predictions.items():
        data[f"{name}_demand"] = forecast["demand"].ravel()
        data[f"{name}_price"] = forecast["price"].ravel()
    pd.DataFrame(data, index=simulation_index).to_parquet(
        os.path.join(args.output, "half_hour_predictions.parquet"))
    day_errors = pd.DataFrame(index=days)
    for name, forecast in predictions.items():
        day_errors[f"{name}_MAPE"] = (100 * np.abs(truth["demand"] - forecast["demand"])
                                     / truth["demand"]).mean(axis=1)
        day_errors[f"{name}_accuracy"] = (truth["price"] == forecast["price"]).mean(axis=1) * 100
    types.join(day_errors).to_csv(os.path.join(args.output, "day_types_and_errors.csv"))
    policies = {}
    chosen = aggregate[(aggregate["period"] == "strategy_selection")
                       & aggregate["model"].str.startswith("adapted_")]
    for task in ("demand", "price"):
        row = chosen.loc[chosen["MAPE"].idxmin()] if task == "demand" else \
            chosen.loc[chosen["balanced_accuracy"].idxmax()]
        strategy = row["model"].removeprefix("adapted_")
        policies[task] = {
            **selected[task], **STRATEGIES[strategy], "strategy": strategy,
            "training_end": "2014-12-31" if strategy == "frozen" else None,
        }
    policies["strategy_selection_end"] = args.selection_end
    policies["quick"] = args.quick
    with open(os.path.join(args.output, "final_config.json"), "w", encoding="utf-8") as handle:
        json.dump(policies, handle, indent=2)
    # Empirical seven-day ranges are a starting point, not a December prediction.
    weekly = []
    for task in ("demand", "price"):
        forecast = predictions[f"adapted_{policies[task]['strategy']}"]
        starts = np.flatnonzero(holdout_mask)[::7]
        for start in starts:
            if start + 7 <= len(days):
                values = metric_values(
                    truth["demand"][start:start + 7].ravel(),
                    truth["price"][start:start + 7].ravel(),
                    forecast["demand"][start:start + 7].ravel(),
                    forecast["price"][start:start + 7].ravel())
                weekly.append({"task": task, "week_start": str(days[start].date()), **values})
    pd.DataFrame(weekly).to_csv(os.path.join(args.output, "holdout_weekly_metrics.csv"), index=False)
    save_figures(args.output, metrics, types, shares, typical, predictions, days, truth)
    with open(os.path.join(args.output, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"seconds": time.monotonic() - started, "fingerprint": fingerprint,
                   "strategy_selection_end": args.selection_end, "bootstrap": args.bootstrap,
                   "reference": "Detectors calibrated on 2014; PCA/clustering fit on 2010-2013",
                   "drift_policy": "Input OR error signal; 180-day retrain cooldown; six updates",
                   "threads": 2}, handle, indent=2)
    print(f"Experiment outputs saved to {args.output} in {time.monotonic() - started:.1f}s",
          flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
