"""Select epochs on 2013, refit through 2013, and evaluate 2014 once."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent / "forecast_package"))
from model import load_data, market_days, training_table
from demand_mlp import DemandMLP, fit_demand_mlp, predict_demand_mlp

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    output = ROOT / "results" / "demand_mlp_early_stopping_2014"
    output.mkdir(parents=True, exist_ok=False)
    columns = json.loads((ROOT / "code" / "demand_feature_columns.json").read_text())
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2015-01-01 00:00:00"]
    features, labels = training_table(frame)
    features = features.loc[:, columns]
    days = market_days(features.index)
    train, select = days.year <= 2012, days.year == 2013
    print("Selecting epochs: train 2010–2012, validate 2013", flush=True)
    selection = fit_demand_mlp(features.loc[train], labels.loc[train, "demand"],
        epochs=100, validation=(features.loc[select], labels.loc[select, "demand"]), patience=10)
    epochs = selection["best_epoch"]
    assert 1 <= epochs <= 100
    pd.DataFrame(selection["history"]).to_csv(output / "selection_history_2013.csv", index=False)
    print(f"Selected {epochs} epochs; refitting 2010–2013", flush=True)
    refit, valid = days.year <= 2013, days.year == 2014
    bundle = fit_demand_mlp(features.loc[refit], labels.loc[refit, "demand"], epochs=epochs)
    predicted = predict_demand_mlp(bundle, features.loc[valid])
    assert np.isfinite(predicted).all()
    actual = labels.loc[valid, "demand"].to_numpy()
    row = {"model": "MLP_selected_epochs", "MAPE": float(np.mean(np.abs(actual-predicted)/actual)*100),
           "MAE": float(np.mean(np.abs(actual-predicted)))}
    previous = pd.read_csv(ROOT / "results" / "demand_mlp_5epoch_2014" / "metrics.csv")
    comparison = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    comparison.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame({"actual": actual, "predicted": predicted}, index=features.index[valid]).to_csv(
        output / "predictions.csv", index_label="timestamp")
    torch.save(bundle["model"].state_dict(), output / "weights.pt")
    joblib.dump({"scaler": bundle["scaler"], "features": columns}, output / "preprocessing.joblib")
    reloaded = DemandMLP(len(columns))
    reloaded.load_state_dict(torch.load(output / "weights.pt", weights_only=True, map_location="cpu"))
    np.testing.assert_allclose(predicted, predict_demand_mlp(dict(bundle, model=reloaded), features.loc[valid]))
    metadata = {"selection_training_years": [2010,2011,2012], "selection_year": 2013,
        "max_epochs": 100, "patience": 10, "selected_epochs": epochs,
        "selection_epochs_run": len(selection["history"]),
        "best_selection_MAPE": selection["best_validation_MAPE"],
        "refit_training_years": [2010,2011,2012,2013], "evaluation_year": 2014,
        "architecture": [405,128,64,1], "device": "cpu", "threads": 2,
        "torch": torch.__version__, "refit_epoch_seconds": bundle["epoch_seconds"],
        "state_dict_reload_passed": True, "holdout_evaluated": False,
        "seconds_total": time.monotonic()-started}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
