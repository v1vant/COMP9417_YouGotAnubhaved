"""Compare matched classifiers with/without out-of-sample price forecasts."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from run_experiments import load_data, market_days, training_table, fit_model

ROOT = Path(__file__).resolve().parents[1]


def regression_features(predictions):
    return predictions[["nsw_price_pred", "nsw_price_margin"]].rename(columns={
        "nsw_price_pred": "regressed_price",
        "nsw_price_margin": "regressed_price_margin",
    })


def main():
    output = ROOT / "results" / "augmented_price_2014"
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "results" / "price_regression_2014"
    with open(source / "metadata.json") as handle:
        metadata = json.load(handle)
    assert not metadata["quick"]
    assert metadata["training_end"] == "2013-12-31"
    config = metadata["config"]

    oof = pd.concat([
        pd.read_csv(
            ROOT / "results" / "price_stacking_2014" / f"oof_{year}.csv",
            index_col="timestamp", parse_dates=True,
        )
        for year in (2011, 2012, 2013)
    ]).sort_index()
    validation = pd.read_csv(
        source / "predictions.csv", index_col="timestamp", parse_dates=True,
    )
    assert oof.index.is_unique
    assert (market_days(oof.index).year < 2014).all()
    assert (market_days(validation.index).year == 2014).all()
    assert oof.index.max() < validation.index.min()

    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2015-01-01 00:00:00"]
    features, labels = training_table(frame)
    assert features.shape[1] == 571
    pd.testing.assert_series_equal(
        labels.loc[oof.index, "price"], oof.actual_up,
        check_names=False, check_dtype=False,
    )
    train_x = features.loc[oof.index]
    valid_x = features.loc[validation.index]
    train_augmented = train_x.join(regression_features(oof))
    valid_augmented = valid_x.join(regression_features(validation))
    assert train_augmented.shape[1] == valid_augmented.shape[1] == 573
    assert not train_augmented.isna().any().any()
    assert not valid_augmented.isna().any().any()

    rows, models = [], {}
    for name, x, v in (
        ("matched_reference", train_x, valid_x),
        ("augmented", train_augmented, valid_augmented),
    ):
        model = fit_model("price", config, x, labels.loc[x.index])
        probability = model["estimator"].predict_proba(
            v.loc[:, model["features"]])[:, 1]
        prediction = probability > config["threshold"]
        validation[f"{name}_probability"] = probability
        rows.append({
            "model": name,
            "accuracy": float((prediction == validation.actual_up).mean()*100),
            "balanced_accuracy": float(balanced_accuracy_score(
                validation.actual_up, prediction)*100),
        })
        models[name] = model
        print(f"Finished {name}", flush=True)

    rows.append({
        "model": "existing_classifier_2010_2013",
        "accuracy": float(
            (validation.classifier_up == validation.actual_up).mean()*100),
        "balanced_accuracy": float(balanced_accuracy_score(
            validation.actual_up, validation.classifier_up)*100),
    })
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    validation.to_csv(output / "predictions.csv", index_label="timestamp")
    joblib.dump(models, output / "validation_models.joblib")
    with open(output / "metadata.json", "w") as handle:
        json.dump({
            "classifier_training_years": [2011, 2012, 2013],
            "validation_year": 2014,
            "base_config": config,
            "added_features": ["regressed_price", "regressed_price_margin"],
            "regression_training_inputs": "Chronological out-of-sample folds",
        }, handle, indent=2)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
