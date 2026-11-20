"""Evaluate the frozen 50/50 demand blend on 2025–October 2026."""
import hashlib
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(actual, predicted):
    return {"MAPE": float(np.mean(np.abs(actual-predicted)/actual)*100),
            "MAE": float(np.mean(np.abs(actual-predicted)))}


def main():
    started = time.monotonic()
    output = ROOT / "results" / "frozen_demand_blend_holdout"
    output.mkdir(parents=True, exist_ok=False)
    columns_path = ROOT / "code" / "demand_feature_columns.json"
    columns = json.loads(columns_path.read_text())
    previous = json.loads((ROOT / "results" / "demand_mlp_10epoch_2015_2024" / "metadata.json").read_text())
    mlp_source = ROOT / "code" / "demand_mlp.py"
    assert previous["epochs"] == 10 and previous["source_sha256"] == digest(mlp_source)
    frozen = {"demand_weights": {"current_LightGBM": 0.5, "MLP": 0.5},
              "MLP": {"architecture": [405,128,64,1], "epochs": 10, "seed": 9417,
                      "batch_size": 1024, "optimizer": "Adam", "learning_rate": 0.001,
                      "loss": "Inverse-demand-weighted absolute residual error",
                      "anchor": "Mean of yesterday and last-week same-slot demand"},
              "initial_training_end": "2023-12-31", "retrain_day": "2026-01-01",
              "retrain_training_end": "2025-12-31", "evaluation_start": "2025-01-01",
              "evaluation_end": "2026-10-31", "feature_schema_sha256": digest(columns_path),
              "MLP_source_sha256": digest(mlp_source), "price": "Unchanged current LightGBM",
              "gate_enabled": False, "tuning_on_this_period": False,
              "note": "Later period already inspected for standalone LightGBM; first blend evaluation."}
    (output / "frozen_config.json").write_text(json.dumps(frozen,indent=2))
    artifacts = ROOT / "code" / "forecast_package" / "artifacts"
    before = {p.name:digest(p) for p in artifacts.iterdir() if p.is_file()}
    reference = ROOT / "results" / "holdout_original_vs_current"
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    features, labels = training_table(frame)
    features = features.loc[:,columns]
    days = market_days(features.index)
    evaluation = days >= pd.Timestamp("2025-01-01")
    index = features.index[evaluation]
    eval_days = days[evaluation]
    assert eval_days.min() == pd.Timestamp("2025-01-01")
    assert eval_days.max() == pd.Timestamp("2026-10-31")
    truth = np.load(reference / "truth.npz")
    np.testing.assert_array_equal(truth["days"],eval_days.unique().to_numpy())
    actual = labels.loc[index,"demand"].to_numpy()
    np.testing.assert_array_equal(actual,truth["demand"].ravel())
    np.testing.assert_array_equal(labels.loc[index,"price"],truth["price"].ravel())
    mlp = pd.Series(index=index,dtype=float)
    audit = []
    for year, end in ((2025,"2023-12-31"),(2026,"2025-12-31")):
        train, test = days <= pd.Timestamp(end), days.year == year
        assert days[train].max() < days[test].min()
        print(f"Fitting ten-epoch MLP for {year}, training through {end}",flush=True)
        bundle = fit_demand_mlp(features.loc[train],labels.loc[train,"demand"],epochs=10)
        predicted = predict_demand_mlp(bundle,features.loc[test])
        mlp.loc[features.index[test]] = predicted
        torch.save(bundle["model"].state_dict(),output / f"weights_{year}.pt")
        joblib.dump({"scaler":bundle["scaler"],"features":columns},output / f"preprocessing_{year}.joblib")
        replay = DemandMLP(len(columns))
        replay.load_state_dict(torch.load(output / f"weights_{year}.pt",weights_only=True,map_location="cpu"))
        np.testing.assert_allclose(predicted,predict_demand_mlp(dict(bundle,model=replay),features.loc[test]))
        audit.append({"forecast_year":year,"training_end":str(days[train].max().date()),
                      "training_rows":int(train.sum()),"epochs":10})
        pd.DataFrame(audit).to_csv(output / "retraining_audit.csv",index=False)
    assert mlp.notna().all() and np.isfinite(mlp.to_numpy()).all()
    lgbm = np.load(reference / "current_demand.npy").ravel()
    predictions = {"current_LightGBM":lgbm,"MLP_10_epochs":mlp.to_numpy(),
                   "frozen_50_50":0.5*lgbm+0.5*mlp.to_numpy(),
                   "baseline_1d":frame.nsw_demand.shift(48).loc[index].to_numpy()}
    totals, periods = [], []
    for name, predicted in predictions.items():
        assert predicted.shape == actual.shape
        totals.append(dict(model=name,**metrics(actual,predicted)))
        for frequency in ("M","Y"):
            period_index = eval_days.to_period(frequency)
            for period in period_index.unique():
                mask = period_index == period
                periods.append(dict(model=name,frequency=frequency,period=str(period),
                                    **metrics(actual[mask],predicted[mask])))
    pd.DataFrame(totals).to_csv(output / "aggregate_metrics.csv",index=False)
    pd.DataFrame(periods).to_csv(output / "period_metrics.csv",index=False)
    pd.DataFrame(dict(actual=actual,**predictions),index=index).to_csv(output / "predictions.csv",index_label="timestamp")
    np.save(output / "blend_demand.npy",predictions["frozen_50_50"].reshape(-1,48))
    y = actual.reshape(-1,48)
    diff = (np.abs(y-lgbm.reshape(-1,48))/y -
            np.abs(y-predictions["frozen_50_50"].reshape(-1,48))/y).mean(axis=1)*100
    rng = np.random.default_rng(9417)
    n = len(diff)
    starts = rng.integers(0,n,size=(5000,int(np.ceil(n/7))))
    indices = ((starts[:,:,None]+np.arange(7))%n).reshape(5000,-1)[:,:n]
    interval = np.percentile(diff[indices].mean(axis=1),[2.5,97.5]).tolist()
    before_map = metrics(actual,lgbm)["MAPE"]
    metadata = {"status":"passed","seconds_total":time.monotonic()-started,
                "MAPE_improvement_pp":float(diff.mean()),
                "relative_MAPE_improvement_percent":float(diff.mean()/before_map*100),
                "paired_7day_bootstrap_CI95":interval,"days":n,
                "state_dict_reload_passed":True,"existing_artifacts_unchanged":True}
    assert before == {p.name:digest(p) for p in artifacts.iterdir() if p.is_file()}
    assert "lightgbm" not in sys.modules
    (output / "completion.json").write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(totals).to_string(index=False),flush=True)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
