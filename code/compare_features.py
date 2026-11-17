"""Compare original/current features on 2015–2024, excluding the holdout."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_experiments import (
    baseline_predictions, fit_model, load_data, market_days,
    metric_values, period_metrics, simulate_policy, training_table,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "results" / "feature_comparison_2015_2024"
    output.mkdir(parents=True, exist_ok=False)
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2025-01-01 00:00:00"]
    features, labels = training_table(frame)

    # Remove only the features introduced during this session.
    added = [
        col for col in features
        if any(part in col for part in (
            "_profile_", "_before_", "_ramp_", "_change_from_open",
        )) or col == "nsw_price_weekly_margin"
    ]
    original = features.drop(columns=added)
    assert original.shape[1] == 240
    assert features.shape[1] == 571

    feature_days = market_days(features.index)
    initial = feature_days.year <= 2014
    evaluation = feature_days >= pd.Timestamp("2015-01-01")
    days = feature_days[evaluation].unique()
    assert days.max() == pd.Timestamp("2024-12-31")
    truth = {
        task: labels.loc[evaluation, task].to_numpy().reshape(-1, 48)
        for task in ("demand", "price")
    }
    with open(ROOT / "results" / "final_config.json") as handle:
        selection = json.load(handle)
    predictions = baseline_predictions(frame, features.index[evaluation])
    retrains = []

    # Periodic retraining does not use detector signals. The existing
    # simulator still requires detector inputs, so supply neutral values.
    input_scores = pd.Series(0.0, index=days)
    reference = np.zeros(365)
    for version, table in (("original", original), ("current", features)):
        predictions[version] = {}
        for task in ("demand", "price"):
            config = selection[task]["config"]
            model = fit_model(
                task, config, table.loc[initial], labels.loc[initial],
            )
            expected = 240 if version == "original" else (
                405 if task == "demand" else 571
            )
            assert len(model["features"]) == expected
            forecast, updates, _ = simulate_policy(
                task, "periodic_expanding", model, config,
                table, labels, days, input_scores, reference, reference,
            )
            predictions[version][task] = forecast
            retrains.extend(dict(version=version, **row) for row in updates)
            np.save(output / f"{version}_{task}.npy", forecast)

    totals, periods = [], []
    for name, forecast in predictions.items():
        totals.append(dict(
            model=name,
            **metric_values(
                truth["demand"].ravel(), truth["price"].ravel(),
                forecast["demand"].ravel(), forecast["price"].ravel(),
            ),
        ))
        table = period_metrics(days, truth, forecast, bootstrap=1000)
        table.insert(0, "model", name)
        periods.append(table)
    aggregate = pd.DataFrame(totals)
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    pd.concat(periods).to_csv(output / "period_metrics.csv", index=False)
    pd.DataFrame(retrains).to_csv(output / "retrains.csv", index=False)
    with open(output / "config.json", "w") as handle:
        json.dump(selection, handle, indent=2)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
