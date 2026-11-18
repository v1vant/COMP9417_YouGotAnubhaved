"""Fixed five-epoch price MLP evaluated on 2014; no LightGBM deserialization."""
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent / "forecast_package"))
from model import load_data, market_days, training_table
from price_mlp import PriceMLP, fit_price_mlp, predict_price_probability

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    output = ROOT / "results" / "price_mlp_5epoch_2014"
    output.mkdir(parents=True, exist_ok=False)
    columns = json.loads((ROOT / "code" / "price_feature_columns.json").read_text())
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2015-01-01 00:00:00"]
    features, labels = training_table(frame)
    features = features.loc[:, columns]
    days = market_days(features.index)
    train, valid = days.year <= 2013, days.year == 2014
    reference_dir = ROOT.parent / "COMP9417_price_regression" / "results" / "price_regression_2014"
    source = json.loads((reference_dir / "metadata.json").read_text())
    assert not source["quick"] and source["training_end"] == "2013-12-31" and source["features"] == 571
    reference = pd.read_csv(reference_dir / "predictions.csv", index_col="timestamp", parse_dates=True)
    assert reference.index.equals(features.index[valid])
    np.testing.assert_array_equal(reference.actual_up, labels.loc[valid, "price"])
    bundle = fit_price_mlp(features.loc[train], labels.loc[train, "price"], epochs=5)
    probability = predict_price_probability(bundle, features.loc[valid])
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    actual = labels.loc[valid, "price"].to_numpy()
    predictions = pd.DataFrame({"actual_up": actual, "mlp_probability": probability,
                               "classifier_probability": reference.classifier_probability}, index=reference.index)
    rows = []
    comparisons = [("MLP_threshold_0.5", probability>0.5),
                   ("MLP_threshold_0.57", probability>0.57),
                   ("current_LightGBM_0.57", reference.classifier_up.to_numpy()),
                   ("baseline_1d", reference.baseline_1d_up.to_numpy()),
                   ("baseline_7d", reference.baseline_7d_up.to_numpy())]
    for name, predicted in comparisons:
        rows.append({"model": name, "accuracy": float(np.mean(actual==predicted)*100),
                     "balanced_accuracy": float(balanced_accuracy_score(actual,predicted)*100)})
    torch.save(bundle["model"].state_dict(), output / "weights.pt")
    joblib.dump({"scaler": bundle["scaler"], "features": columns}, output / "preprocessing.joblib")
    reloaded = PriceMLP(len(columns))
    reloaded.load_state_dict(torch.load(output / "weights.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(probability, predict_price_probability(dict(bundle,model=reloaded),features.loc[valid]))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index_label="timestamp")
    metadata = {"training_years": [2010,2011,2012,2013], "validation_year": 2014,
                "architecture": [571,128,64,1], "epochs": 5, "batch_size": 1024,
                "loss": "Class-balanced binary cross-entropy with logits",
                "class_weights": bundle["class_weights"], "history": bundle["history"],
                "device": "cpu", "threads": 2, "torch": torch.__version__,
                "python": platform.python_version(), "seconds_total": time.monotonic()-started,
                "thresholds_fixed": [0.5,0.57], "state_dict_reload_passed": True,
                "holdout_evaluated": False}
    (output / "metadata.json").write_text(json.dumps(metadata,indent=2))
    assert "lightgbm" not in sys.modules
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps(metadata,indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
