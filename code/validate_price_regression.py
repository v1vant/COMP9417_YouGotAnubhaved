"""Train price regression/classification on 2010–2013 and compare on 2014."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from run_experiments import load_data, market_days, training_table, fit_model
from price_regression import fit_price_regressor, predict_price_regressor

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="120 trees; smoke test only")
    args = parser.parse_args()
    output = ROOT / "results" / ("price_regression_2014_quick" if args.quick else "price_regression_2014")
    output.mkdir(parents=True, exist_ok=False)
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2015-01-01 00:00:00"]
    features, labels = training_table(frame)  # Weather shape remains disabled.
    assert features.shape[1] == 571
    days = market_days(features.index)
    train, valid = days.year <= 2013, days.year == 2014
    config = {"family": "boosted", "trees": 120 if args.quick else 600,
              "leaves": 31, "learning_rate": 0.04, "threshold": 0.57}
    regressor = fit_price_regressor(features.loc[train],
                                    frame.loc[features.index[train], "nsw_price"], config)
    means = frame["nsw_price"].groupby(market_days(frame.index)).mean().shift(1)
    threshold = pd.Series(days[valid], index=features.index[valid]).map(means)
    predictions = predict_price_regressor(regressor, features.loc[valid], threshold)
    classifier = fit_model("price", config, features.loc[train], labels.loc[train])
    probability = classifier["estimator"].predict_proba(
        features.loc[valid, classifier["features"]])[:, 1]
    predictions["classifier_probability"] = probability
    predictions["classifier_up"] = (probability > config["threshold"]).astype(int)
    predictions["actual_price"] = frame.loc[predictions.index, "nsw_price"]
    predictions["previous_day_mean"] = threshold
    predictions["actual_up"] = labels.loc[valid, "price"]
    for lag in (1, 7):
        historical = frame["nsw_price"].shift(48 * lag).loc[predictions.index]
        predictions[f"baseline_{lag}d_up"] = (historical > threshold).astype(int)
    rows = []
    for name, column in (("regressor", "nsw_price_up"), ("classifier", "classifier_up"),
                         ("baseline_1d", "baseline_1d_up"), ("baseline_7d", "baseline_7d_up")):
        rows.append({"model": name,
                     "accuracy": float((predictions[column] == predictions.actual_up).mean()*100),
                     "balanced_accuracy": float(balanced_accuracy_score(
                         predictions.actual_up, predictions[column])*100)})
    errors = predictions.nsw_price_pred - predictions.actual_price
    metadata = {"training_end": "2013-12-31", "validation_year": 2014,
                "features": len(features.columns), "quick": args.quick, "config": config,
                "price_MAE": float(errors.abs().mean()),
                "price_RMSE": float(np.sqrt(np.mean(errors**2))),
                "note": "No ensemble or probability calibration fitted; holdout not evaluated."}
    predictions.to_csv(output / "predictions.csv", index_label="timestamp")
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    with open(output / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    joblib.dump({"regressor": regressor, "classifier": classifier}, output / "validation_models.joblib")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
