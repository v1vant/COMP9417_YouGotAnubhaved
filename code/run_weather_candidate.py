"""Validate optional weather shapes, then simulate promising features before 2025."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_experiments import (
    fit_model, load_data, market_days, metric_values, period_metrics,
    predict_task, simulate_policy, training_table,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "results" / "weather_candidate"
    output.mkdir(parents=True, exist_ok=False)
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2025-01-01 00:00:00"]
    candidate, labels = training_table(frame, weather_shape=True)
    # Preserve the reference feature order, which matters for tree sampling.
    reference, reference_labels = training_table(frame)
    assert candidate.index.equals(reference.index)
    pd.testing.assert_frame_equal(labels, reference_labels)
    assert reference.shape[1] == 571 and candidate.shape[1] == 597
    with open(ROOT / "results" / "final_config.json") as handle:
        selection = json.load(handle)
    with open(ROOT / "results" / "feature_comparison_2015_2024" / "config.json") as handle:
        assert selection == json.load(handle)
    with open(output / "config.json", "w") as handle:
        json.dump(selection, handle, indent=2)
    days_all = market_days(candidate.index)
    train, valid = days_all.year <= 2013, days_all.year == 2014
    rows = []
    for version, features in (("reference", reference), ("weather", candidate)):
        forecasts = {}
        for task in ("demand", "price"):
            model = fit_model(task, selection[task]["config"],
                              features.loc[train], labels.loc[train])
            expected = (405 if task == "demand" else 571) + (26 if version == "weather" else 0)
            assert len(model["features"]) == expected
            forecasts[task] = predict_task(model, features.loc[valid])
        row = dict(version=version, **metric_values(
            labels.loc[valid, "demand"], labels.loc[valid, "price"],
            forecasts["demand"], forecasts["price"],
        ))
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "validation_2014.csv", index=False)
        print("2014 validation:", row, flush=True)
    promising = (rows[1]["MAPE"] < rows[0]["MAPE"] or
                 rows[1]["balanced_accuracy"] > rows[0]["balanced_accuracy"])
    with open(output / "validation_decision.json", "w") as handle:
        json.dump({"promising": promising,
                   "rule": "Continue if either task improves; fixed configurations and thresholds."},
                  handle, indent=2)
    if not promising:
        print("Neither task improved: stopping before historical simulation.", flush=True)
        return

    print("Promising validation: starting 2015–2024 simulation.", flush=True)
    evaluation = days_all >= pd.Timestamp("2015-01-01")
    days = days_all[evaluation].unique()
    assert days.max() == pd.Timestamp("2024-12-31")
    truth = {task: labels.loc[evaluation, task].to_numpy().reshape(-1, 48)
             for task in ("demand", "price")}
    forecasts, retrains = {}, []
    for task in ("demand", "price"):
        initial = days_all.year <= 2014
        config = selection[task]["config"]
        model = fit_model(task, config, candidate.loc[initial], labels.loc[initial])
        # Detector inputs are unused by the periodic retraining policy.
        forecasts[task], updates, _ = simulate_policy(
            task, "periodic_expanding", model, config, candidate, labels,
            days, pd.Series(0.0, index=days), np.zeros(365), np.zeros(365),
        )
        retrains.extend(updates)
        np.save(output / f"weather_{task}.npy", forecasts[task])
        pd.DataFrame(retrains).to_csv(output / "retrains.csv", index=False)
        print("Finished historical", task, flush=True)

    totals, periods = [], []
    reference_dir = ROOT / "results" / "feature_comparison_2015_2024"
    reference_forecasts = {
        task: np.load(reference_dir / f"current_{task}.npy")
        for task in ("demand", "price")
    }
    for version, predictions in (("reference", reference_forecasts), ("weather", forecasts)):
        assert all(predictions[task].shape == truth[task].shape for task in truth)
        totals.append(dict(version=version, **metric_values(
            truth["demand"].ravel(), truth["price"].ravel(),
            predictions["demand"].ravel(), predictions["price"].ravel(),
        )))
        table = period_metrics(days, truth, predictions, bootstrap=1000)
        table.insert(0, "version", version)
        periods.append(table)
    pd.DataFrame(totals).to_csv(output / "aggregate_metrics.csv", index=False)
    pd.concat(periods).to_csv(output / "period_metrics.csv", index=False)
    print(pd.DataFrame(totals).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
