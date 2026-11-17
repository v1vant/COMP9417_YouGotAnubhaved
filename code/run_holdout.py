"""Evaluate frozen original/current feature choices on the 2025–2026 holdout."""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_experiments import (
    baseline_predictions, fit_model, load_data, market_days, metric_values,
    period_metrics, simulate_policy, training_table,
)

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    output = ROOT / "results" / "holdout_original_vs_current"
    output.mkdir(parents=True, exist_ok=False)
    artifacts = ROOT / "code" / "forecast_package" / "artifacts"
    before = {p.name: digest(p) for p in artifacts.iterdir() if p.is_file()}
    data_path = ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv"
    frame = load_data(data_path)
    features, labels = training_table(frame)  # Weather candidate stays disabled.
    added = [
        col for col in features
        if any(part in col for part in (
            "_profile_", "_before_", "_ramp_", "_change_from_open",
        )) or col == "nsw_price_weekly_margin"
    ]
    original = features.drop(columns=added)
    assert original.shape[1] == 240 and features.shape[1] == 571
    feature_days = market_days(features.index)
    initial = feature_days <= pd.Timestamp("2023-12-31")
    evaluation = feature_days >= pd.Timestamp("2025-01-01")
    days = feature_days[evaluation].unique()
    assert days[0] == pd.Timestamp("2025-01-01")
    assert days[-1] == pd.Timestamp("2026-10-31")
    assert feature_days[initial].max() == pd.Timestamp("2023-12-31")
    truth = {task: labels.loc[evaluation, task].to_numpy().reshape(-1, 48)
             for task in ("demand", "price")}
    np.savez(output / "truth.npz", days=days.to_numpy(), **truth)
    with open(ROOT / "results" / "final_config.json") as handle:
        selection = json.load(handle)
    with open(ROOT / "results" / "feature_comparison_2015_2024" / "config.json") as handle:
        assert selection == json.load(handle), "Configuration changed after pre-holdout comparison"
    metadata = {
        "config": selection, "python": platform.python_version(),
        "initial_training_end": "2023-12-31", "scheduled_retrain": "2026-01-01",
        "evaluation_start": str(days[0].date()), "evaluation_end": str(days[-1].date()),
        "training_start": str(features.index[initial][0]), "data_sha256": digest(data_path),
        "artifact_hashes_before": before,
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in (
            ROOT / "code" / "forecast_package" / "model.py",
            ROOT / "code" / "forecast_package" / "train.py",
            ROOT / "code" / "run_experiments.py", Path(__file__),
        )},
    }
    with open(output / "frozen_config.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    predictions = baseline_predictions(frame, features.index[evaluation])
    retrains = []
    for version, table in (("original", original), ("current", features)):
        predictions[version] = {}
        for task in ("demand", "price"):
            task_started = time.monotonic()
            config = selection[task]["config"]
            print(f"Starting {version}/{task}", flush=True)
            model = fit_model(task, config, table.loc[initial], labels.loc[initial])
            expected = 240 if version == "original" else (405 if task == "demand" else 571)
            assert len(model["features"]) == expected
            forecast, updates, _ = simulate_policy(
                task, "periodic_expanding", model, config, table, labels,
                days, pd.Series(0.0, index=days), np.zeros(365), np.zeros(365),
            )
            assert len(updates) == 1 and updates[0]["effective_day"] == "2026-01-01"
            assert updates[0]["training_end"] == "2025-12-31"
            predictions[version][task] = forecast
            retrains.extend(dict(version=version, **row) for row in updates)
            np.save(output / f"{version}_{task}.npy", forecast)
            pd.DataFrame(retrains).to_csv(output / "retrains.csv", index=False)
            print(f"Finished {version}/{task} in {time.monotonic()-task_started:.1f}s", flush=True)

    totals, periods = [], []
    for name, forecast in predictions.items():
        totals.append(dict(model=name, **metric_values(
            truth["demand"].ravel(), truth["price"].ravel(),
            forecast["demand"].ravel(), forecast["price"].ravel(),
        )))
        table = period_metrics(days, truth, forecast, bootstrap=1000)
        table.insert(0, "model", name)
        periods.append(table)
    aggregate = pd.DataFrame(totals)
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    pd.concat(periods).to_csv(output / "period_metrics.csv", index=False)
    after = {p.name: digest(p) for p in artifacts.iterdir() if p.is_file()}
    assert before == after, "Current artifacts changed during evaluation"
    with open(output / "completion.json", "w") as handle:
        json.dump({"status": "passed", "seconds": time.monotonic()-started,
                   "current_artifacts_unchanged": True}, handle, indent=2)
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
