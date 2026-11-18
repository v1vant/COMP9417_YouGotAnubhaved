"""Select KNN settings on 2013; evaluate the frozen winner on 2014."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent / "forecast_package"))
from model import load_data, market_days, training_table
from price_knn import fit_price_knn, predict_price_knn

ROOT = Path(__file__).resolve().parents[1]


def scores(actual, probability):
    predicted = probability > 0.5
    return {
        "accuracy": float(np.mean(actual == predicted)*100),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)*100),
    }


def main():
    started = time.monotonic()
    output = ROOT / "results" / "price_knn_2014"
    output.mkdir(parents=True, exist_ok=False)
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2015-01-01 00:00:00"]
    features, labels = training_table(frame)
    days = market_days(features.index)
    train = (days.year >= 2011) & (days.year <= 2012)
    select = days.year == 2013
    candidates = []
    for feature_set in ("price_time", "full"):
        for neighbors in (25, 75, 150):
            model = fit_price_knn(
                features.loc[train], labels.loc[train, "price"],
                neighbors=neighbors, feature_set=feature_set,
            )
            probability = predict_price_knn(model, features.loc[select])
            row = {
                "feature_set": feature_set, "neighbors": neighbors,
                **scores(labels.loc[select, "price"].to_numpy(), probability),
            }
            candidates.append(row)
            pd.DataFrame(candidates).to_csv(output / "selection_2013.csv", index=False)
            print(row, flush=True)
    best = max(candidates, key=lambda row: row["balanced_accuracy"])
    (output / "selected_config.json").write_text(json.dumps(best, indent=2))

    refit = (days.year >= 2012) & (days.year <= 2013)
    valid = days.year == 2014
    model = fit_price_knn(
        features.loc[refit], labels.loc[refit, "price"],
        neighbors=best["neighbors"], feature_set=best["feature_set"],
    )
    probability = predict_price_knn(model, features.loc[valid])
    result = scores(labels.loc[valid, "price"].to_numpy(), probability)
    pd.DataFrame([dict(model="KNN", **result)]).to_csv(
        output / "metrics.csv", index=False)
    pd.DataFrame({
        "probability": probability,
        "actual_up": labels.loc[valid, "price"],
    }, index=features.index[valid]).to_csv(
        output / "predictions.csv", index_label="timestamp")
    joblib.dump(model, output / "model.joblib")
    metadata = {"selection_training_years": [2011,2012], "selection_year": 2013,
                "final_training_years": [2012,2013], "evaluation_year": 2014,
                "selected": best, "threshold": 0.5, "seconds": time.monotonic()-started,
                "holdout_evaluated": False}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("2014:", result, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2), config_context(working_memory=256):
        main()
