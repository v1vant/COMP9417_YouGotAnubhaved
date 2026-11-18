"""Chronological quantile-price augmentation, evaluated only on 2014."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_pinball_loss
from threadpoolctl import threadpool_limits

from run_experiments import load_data, market_days, training_table, fit_model
from price_regression import fit_price_regressor

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = (0.1, 0.5, 0.9)


def quantile_features(predictions, threshold):
    assert predictions.index.equals(threshold.index)
    # Independent quantile models can cross; monotonically rearrange per row.
    values = np.sort(predictions[["raw_q10", "raw_q50", "raw_q90"]].to_numpy(), axis=1)
    low, median, high = values.T
    return pd.DataFrame({
        "price_q10_margin": low-threshold.to_numpy(),
        "price_q50_margin": median-threshold.to_numpy(),
        "price_q90_margin": high-threshold.to_numpy(),
        "price_q80_width": high-low,
        "price_quantile_asymmetry": high+low-2*median,
    }, index=predictions.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = ROOT / "results" / "quantile_price_2014"
    output.mkdir(parents=True, exist_ok=args.resume)
    started = time.monotonic()
    with open(ROOT / "results" / "price_regression_2014" / "metadata.json") as handle:
        source = json.load(handle)
    assert not source["quick"] and source["training_end"] == "2013-12-31"
    config = source["config"]
    fingerprint = {"config": config, "quantiles": list(QUANTILES), "source_hashes": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), ROOT / "code" / "price_regression.py",
                     ROOT / "code" / "forecast_package" / "model.py",
                     ROOT / "code" / "forecast_package" / "train.py")}}
    frozen = output / "frozen_config.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == fingerprint, "Resume configuration changed"
    else:
        frozen.write_text(json.dumps(fingerprint, indent=2))
    data_path = ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv"
    frame = load_data(data_path).loc[:"2015-01-01 00:00:00"]
    features, labels = training_table(frame)
    assert features.shape[1] == 571
    days = market_days(features.index)
    means = frame["nsw_price"].groupby(market_days(frame.index)).mean().shift(1)
    predictions, audit = [], []
    for year in (2011, 2012, 2013, 2014):
        train, test = days.year < year, days.year == year
        assert days[train].max() < days[test].min()
        fold = pd.DataFrame(index=features.index[test])
        for q in QUANTILES:
            name = f"raw_q{int(q*100)}"
            path = output / f"fold_{year}_{name}.csv"
            print(f"Year {year}, quantile {q}: starting", flush=True)
            if args.resume and path.exists():
                cached = pd.read_csv(path, index_col="timestamp", parse_dates=True)
                assert cached.index.equals(fold.index)
                fold[name] = cached[name]
            else:
                model = fit_price_regressor(features.loc[train],
                    frame.loc[features.index[train], "nsw_price"], config, quantile=q)
                fold[name] = model["estimator"].predict(features.loc[test, model["features"]])
                fold[[name]].to_csv(path, index_label="timestamp")
                if year == 2014:
                    joblib.dump(model, output / f"validation_{name}.joblib")
            print(f"Year {year}, quantile {q}: saved", flush=True)
        assert np.isfinite(fold.to_numpy()).all()
        threshold = pd.Series(days[test], index=fold.index).map(means)
        predictions.append(quantile_features(fold, threshold))
        actual = frame.loc[fold.index, "nsw_price"].to_numpy()
        ordered = np.sort(fold.to_numpy(), axis=1)
        row = {"forecast_year": year, "training_end": str(days[train].max().date()),
               "crossing_percent": float(np.mean((np.diff(fold.to_numpy(), axis=1)<0).any(axis=1))*100),
               "interval_coverage_percent": float(np.mean((actual>=ordered[:,0]) & (actual<=ordered[:,2]))*100)}
        row.update({f"pinball_{q}": float(mean_pinball_loss(actual, fold[f"raw_q{int(q*100)}"], alpha=q)) for q in QUANTILES})
        audit.append(row)
        pd.DataFrame(audit).to_csv(output / "quantile_diagnostics.csv", index=False)

    generated = pd.concat(predictions).sort_index()
    augmented = features.loc[generated.index].join(generated)
    assert augmented.shape[1] == 576 and np.isfinite(augmented.to_numpy()).all()
    aug_days = market_days(augmented.index)
    train, valid = aug_days.year < 2014, aug_days.year == 2014
    model = fit_model("price", config, augmented.loc[train], labels.loc[augmented.index[train]])
    probability = model["estimator"].predict_proba(augmented.loc[valid, model["features"]])[:,1]
    actual_up = labels.loc[augmented.index[valid], "price"]
    row = {"model": "quantile_augmented", "accuracy": float(np.mean((probability>config["threshold"])==actual_up)*100),
           "balanced_accuracy": float(balanced_accuracy_score(actual_up, probability>config["threshold"])*100)}
    comparison = pd.read_csv(ROOT / "results" / "augmented_price_2014" / "metrics.csv")
    comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True)
    comparison.to_csv(output / "metrics.csv", index=False)
    result = generated.loc[augmented.index[valid]].copy()
    result["probability"] = probability
    result["predicted_up"] = (probability>config["threshold"]).astype(int)
    result["actual_up"] = actual_up
    result.to_csv(output / "predictions_2014.csv", index_label="timestamp")
    joblib.dump(model, output / "augmented_classifier.joblib")
    (output / "completion.json").write_text(json.dumps({"status": "passed",
        "seconds": time.monotonic()-started, "classifier_training_years": [2011,2012,2013],
        "validation_year": 2014, "holdout_evaluated": False}, indent=2))
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
