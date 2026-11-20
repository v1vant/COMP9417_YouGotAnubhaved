"""Five-epoch CPU timing and 2014 demand validation; no model selection."""
import json
import platform
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
    output = ROOT / "results" / "demand_mlp_5epoch_2014"
    output.mkdir(parents=True, exist_ok=False)
    # Keep LightGBM artifact loading out of the PyTorch process.
    columns = json.loads((ROOT / "code" / "demand_feature_columns.json").read_text())
    assert len(columns) == 405
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    features, labels = training_table(frame)
    features = features.loc[:, columns]
    days = market_days(features.index)
    train, valid = days.year <= 2013, days.year == 2014
    bundle = fit_demand_mlp(features.loc[train], labels.loc[train, "demand"], epochs=5)
    actual = labels.loc[valid, "demand"].to_numpy()
    predictions = predict_demand_mlp(bundle, features.loc[valid])
    assert np.isfinite(predictions).all()
    rows = []
    anchor = features.loc[valid, "demand_anchor"].to_numpy()
    for name, predicted in (("demand_anchor", anchor), ("MLP_5_epochs", predictions)):
        rows.append({"model": name, "MAPE": float(np.mean(np.abs(actual-predicted)/actual)*100),
                     "MAE": float(np.mean(np.abs(actual-predicted)))})
    reference = pd.read_csv(ROOT / "results" / "full_profiles_2014" / "validation_metrics.csv")
    reference = reference.loc[(reference.version == "full_profiles_405") &
                              (reference.task == "demand") & (reference.trees == 900)].iloc[0]
    rows.append({"model": "current_LightGBM_saved_reference", "MAPE": float(reference.MAPE), "MAE": float(reference.MAE)})
    torch.save(bundle["model"].state_dict(), output / "weights.pt")
    joblib.dump({"scaler": bundle["scaler"], "features": columns}, output / "preprocessing.joblib")
    # Confirm the documented CPU state_dict loading path reproduces inference.
    reloaded = DemandMLP(405)
    reloaded.load_state_dict(torch.load(output / "weights.pt", weights_only=True, map_location="cpu"))
    replay = dict(bundle, model=reloaded)
    np.testing.assert_allclose(predictions, predict_demand_mlp(replay, features.loc[valid]))
    epoch_seconds = float(np.mean(bundle["epoch_seconds"][1:]))
    projected_epoch = epoch_seconds * len(features) / int(train.sum())
    metadata = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                "threads": 2, "architecture": [405,128,64,1], "epochs": 5,
                "parameters": sum(p.numel() for p in bundle["model"].parameters()),
                "training_years": [2010,2011,2012,2013], "validation_year": 2014,
                "training_rows": int(train.sum()), "full_data_rows": len(features),
                "epoch_seconds": bundle["epoch_seconds"],
                "projected_full_data_100_epoch_minutes": projected_epoch*100/60,
                "projection_note": "Linear scaling from measured pre-2014 CPU epochs; excludes data preparation and validation overhead. Not a full-data benchmark.",
                "seconds_total": time.monotonic()-started,
                "state_dict_reload_passed": True, "holdout_evaluated": False,
                "note": "Five-epoch feasibility benchmark; no early stopping or hyperparameter search."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame({"actual": actual, "mlp": predictions, "anchor": anchor},
                 index=features.index[valid]).to_csv(output / "predictions.csv", index_label="timestamp")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
