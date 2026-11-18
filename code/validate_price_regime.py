"""Compare current price features with six previous-day regime summaries."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from run_experiments import load_data, market_days, training_table, fit_model, predict_task

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    output = ROOT / "results" / "price_regime_2014"
    output.mkdir(parents=True, exist_ok=False)
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2015-01-01 00:00:00"]
    features, labels = training_table(frame)
    additions = [col for col in features if col.startswith("nsw_price_regime_")]
    assert len(additions) == 6 and features.shape[1] == 577
    reference = features.drop(columns=additions)
    days = market_days(features.index)
    train, valid = days.year <= 2013, days.year == 2014
    with open(ROOT / "results" / "final_config.json") as handle:
        config = json.load(handle)["price"]["config"]
    rows, models = [], {}
    predictions = pd.DataFrame({"actual_up": labels.loc[valid, "price"]})
    for name, table in (("current", reference), ("price_regime", features)):
        model = fit_model("price", config, table.loc[train], labels.loc[train])
        probability = model["estimator"].predict_proba(table.loc[valid, model["features"]])[:, 1]
        prediction = predict_task(model, table.loc[valid])
        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_up"] = prediction
        rows.append({"model": name, "features": len(model["features"]),
                     "accuracy": float(np.mean(prediction == labels.loc[valid, "price"])*100),
                     "balanced_accuracy": float(balanced_accuracy_score(
                         labels.loc[valid, "price"], prediction)*100)})
        models[name] = model
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        print(rows[-1], flush=True)
    predictions.to_csv(output / "predictions.csv", index_label="timestamp")
    joblib.dump(models, output / "validation_models.joblib")
    (output / "metadata.json").write_text(json.dumps({
        "training_years": [2010, 2011, 2012, 2013], "validation_year": 2014,
        "config": config, "added_features": additions,
        "seconds": time.monotonic()-started, "holdout_evaluated": False,
        "note": "Fixed settings and decision threshold, no regression/quantile inputs.",
    }, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
