"""Select gate on 2015–2020; check frozen predictions on 2021–2024."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "forecast_package"))
from model import load_data, market_days, training_table
from demand_gated_blend import gate_features, fit_gate, predict_gate

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "results" / "demand_gated_blend"
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "results" / "demand_mlp_10epoch_2015_2024"
    truth = np.load(source / "truth.npz")
    index = pd.DatetimeIndex(truth["timestamps"])
    actual = truth["demand"]
    mlp = np.load(source / "mlp_demand.npy").ravel()
    lgbm = np.load(
        ROOT / "results" / "feature_comparison_2015_2024" /
        "current_demand.npy").ravel()
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2025-01-01 00:00:00"]
    features, labels = training_table(frame)
    np.testing.assert_array_equal(labels.loc[index, "demand"], actual)
    context = gate_features(features.loc[index], lgbm, mlp)
    days = market_days(index)
    train, check = days.year <= 2020, days.year >= 2021
    assert days[train].max() < days[check].min()
    gate = fit_gate(context.loc[train], lgbm[train], mlp[train], actual[train])
    predicted, weights = predict_gate(gate, context, lgbm, mlp)

    rows = []
    for period, mask in (("selection_2015_2020", train),
                         ("check_2021_2024", check)):
        for name, values in (
            ("LightGBM", lgbm), ("MLP_10", mlp),
            ("50_50", (lgbm+mlp)/2), ("gated", predicted),
        ):
            error = np.abs(actual[mask]-values[mask])
            rows.append({
                "period": period, "model": name,
                "MAPE": float(np.mean(error/actual[mask])*100),
                "MAE": float(error.mean()),
            })
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame({
        "actual": actual, "gated": predicted,
        "LightGBM_weight": weights,
    }, index=index).to_csv(output / "predictions.csv", index_label="timestamp")
    joblib.dump(gate, output / "gate.joblib")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
