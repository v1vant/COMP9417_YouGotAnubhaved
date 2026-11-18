"""Fit stacking on chronological 2011–2013 predictions; evaluate on 2014."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from run_experiments import load_data, market_days, training_table, fit_model
from price_regression import fit_price_regressor, predict_price_regressor
from price_stacking import fit_stacker, predict_stacker

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "results" / "price_regression_2014"
    with open(source / "metadata.json") as handle:
        metadata = json.load(handle)
    assert not metadata["quick"]
    assert metadata["training_end"] == "2013-12-31"
    assert metadata["validation_year"] == 2014
    config = metadata["config"]
    output = ROOT / "results" / "price_stacking_2014"
    output.mkdir(parents=True, exist_ok=False)

    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2014-01-01 00:00:00"]
    features, labels = training_table(frame)
    assert features.shape[1] == metadata["features"] == 571
    days = market_days(features.index)
    means = frame["nsw_price"].groupby(market_days(frame.index)).mean().shift(1)
    folds, fold_records = [], []
    for year in (2011, 2012, 2013):
        train, test = days.year < year, days.year == year
        assert days[train].max() < days[test].min()
        regressor = fit_price_regressor(
            features.loc[train],
            frame.loc[features.index[train], "nsw_price"], config,
        )
        classifier = fit_model(
            "price", config, features.loc[train], labels.loc[train],
        )
        threshold = pd.Series(days[test], index=features.index[test]).map(means)
        fold = predict_price_regressor(regressor, features.loc[test], threshold)
        fold["classifier_probability"] = classifier["estimator"].predict_proba(
            features.loc[test, classifier["features"]])[:, 1]
        fold["actual_up"] = labels.loc[test, "price"]
        folds.append(fold)
        fold.to_csv(output / f"oof_{year}.csv", index_label="timestamp")
        fold_records.append({
            "forecast_year": year,
            "training_end": str(days[train].max().date()),
            "rows": len(fold),
        })
        print(f"Completed out-of-sample fold {year}", flush=True)

    oof = pd.concat(folds).sort_index()
    stacker = fit_stacker(
        oof.classifier_probability, oof.nsw_price_margin, oof.actual_up,
    )
    validation = pd.read_csv(
        source / "predictions.csv", index_col="timestamp", parse_dates=True,
    )
    assert (market_days(validation.index).year == 2014).all()
    result = predict_stacker(
        stacker, validation.classifier_probability, validation.nsw_price_margin,
    )
    validation["stacked_probability"] = result.stacked_probability
    validation["stacked_up"] = result.nsw_price_up
    rows = []
    for name, column in (
        ("classifier", "classifier_up"),
        ("regressor", "nsw_price_up"),
        ("stacker", "stacked_up"),
    ):
        rows.append({
            "model": name,
            "accuracy": float(
                (validation[column] == validation.actual_up).mean() * 100),
            "balanced_accuracy": float(balanced_accuracy_score(
                validation.actual_up, validation[column]) * 100),
        })
    validation.to_csv(output / "predictions.csv", index_label="timestamp")
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    joblib.dump(stacker, output / "stacker.joblib")
    with open(output / "metadata.json", "w") as handle:
        json.dump({"base_config": config, "folds": fold_records,
                   "stacker_threshold": 0.5}, handle, indent=2)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
