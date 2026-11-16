"""Fixed-epoch demand MLP simulation over 2015–2024; no LightGBM loading."""
import argparse
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
from demand_mlp import fit_demand_mlp, predict_demand_mlp

ROOT = Path(__file__).resolve().parents[1]


def metrics(actual, predicted):
    return {"MAPE": float(np.mean(np.abs(actual-predicted)/actual)*100),
            "MAE": float(np.mean(np.abs(actual-predicted)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    started = time.monotonic()
    output = ROOT / "results" / f"demand_mlp_{args.epochs}epoch_2015_2024"
    output.mkdir(parents=True, exist_ok=False)
    columns = json.loads((ROOT / "code" / "demand_feature_columns.json").read_text())
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2025-01-01 00:00:00"]
    features, labels = training_table(frame)
    features = features.loc[:, columns]
    assert features.shape[1] == 405
    days = market_days(features.index)
    evaluation = days >= pd.Timestamp("2015-01-01")
    index = features.index[evaluation]
    eval_days = days[evaluation]
    assert eval_days.min() == pd.Timestamp("2015-01-01")
    assert eval_days.max() == pd.Timestamp("2024-12-31")
    actual = labels.loc[index, "demand"].to_numpy()
    predicted = pd.Series(index=index, dtype=float)
    updates = []
    bundle = None
    for year in range(2015, 2025):
        if bundle is None or year in (2016, 2018, 2020, 2022, 2024):
            # The initial model is fitted through 2014, then biennially.
            train = days < pd.Timestamp(f"{year}-01-01")
            print(f"Fitting MLP for {year}; data through {days[train].max().date()}", flush=True)
            bundle = fit_demand_mlp(features.loc[train], labels.loc[train, "demand"], epochs=args.epochs)
            audit = {"effective_day": f"{year}-01-01", "training_start": str(days[train].min().date()),
                     "training_end": str(days[train].max().date()), "training_rows": int(train.sum()),
                     "epochs": args.epochs, "training_seconds": sum(bundle["epoch_seconds"])}
            updates.append(audit)
            torch.save(bundle["model"].state_dict(), output / f"weights_{year}.pt")
            joblib.dump({"scaler": bundle["scaler"], "features": columns},
                        output / f"preprocessing_{year}.joblib")
            pd.DataFrame(updates).to_csv(output / "retraining_audit.csv", index=False)
        test = days.year == year
        assert days[train].max() < days[test].min()
        values = predict_demand_mlp(bundle, features.loc[test])
        assert np.isfinite(values).all()
        predicted.loc[features.index[test]] = values
        yearly = metrics(labels.loc[test, "demand"].to_numpy(), values)
        pd.DataFrame({"predicted": values}, index=features.index[test]).to_csv(
            output / f"predictions_{year}.csv", index_label="timestamp")
        print(f"Completed {year}: {yearly}", flush=True)
    assert predicted.notna().all()
    forecasts = {f"MLP_{args.epochs}_epochs": predicted.to_numpy()}
    five_epoch_path = ROOT / "results" / "demand_mlp_5epoch_2015_2024"
    if args.epochs != 5 and (five_epoch_path / "mlp_demand.npy").exists():
        earlier_truth = np.load(five_epoch_path / "truth.npz")
        np.testing.assert_array_equal(earlier_truth["timestamps"], index.to_numpy())
        np.testing.assert_array_equal(earlier_truth["demand"], actual)
        forecasts["MLP_5_epochs"] = np.load(five_epoch_path / "mlp_demand.npy").ravel()
    reference = ROOT / "results" / "feature_comparison_2015_2024"
    for name, filename in (("current_LightGBM", "current_demand.npy"),
                           ("original_LightGBM", "original_demand.npy")):
        forecasts[name] = np.load(reference / filename).ravel()
        assert forecasts[name].shape == actual.shape
    for lag in (1, 7):
        forecasts[f"baseline_{lag}d"] = frame.nsw_demand.shift(48*lag).loc[index].to_numpy()
    totals, periods = [], []
    for name, values in forecasts.items():
        totals.append(dict(model=name, **metrics(actual, values)))
        for frequency in ("M", "Y"):
            period_index = eval_days.to_period(frequency)
            for period in period_index.unique():
                mask = period_index == period
                periods.append(dict(model=name, frequency=frequency, period=str(period),
                                    **metrics(actual[mask], values[mask])))
    pd.DataFrame(totals).to_csv(output / "aggregate_metrics.csv", index=False)
    pd.DataFrame(periods).to_csv(output / "period_metrics.csv", index=False)
    np.save(output / "mlp_demand.npy", predicted.to_numpy().reshape(-1,48))
    np.savez(output / "truth.npz", timestamps=index.to_numpy(), demand=actual)
    metadata = {"epochs": args.epochs, "architecture": [405,128,64,1], "seed": 9417,
                "device": "cpu", "threads": 2, "torch": torch.__version__,
                "evaluation_start": "2015-01-01", "evaluation_end": "2024-12-31",
                "holdout_evaluated": False, "seconds_total": time.monotonic()-started,
                "source_sha256": hashlib.sha256((ROOT / "code" / "demand_mlp.py").read_bytes()).hexdigest()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(totals).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
