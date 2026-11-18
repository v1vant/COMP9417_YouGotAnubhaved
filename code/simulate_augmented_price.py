"""Compare augmented/reference price classifiers over 2015–2024 only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from run_experiments import load_data, market_days, training_table, fit_model, simulate_policy
from price_regression import fit_price_regressor, predict_price_regressor

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def price_metrics(actual, predicted):
    return {"accuracy": float(np.mean(actual == predicted)*100),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)*100)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-results", type=Path, default=(
        ROOT.parent / "COMP9417_YouGotAnubhaved" / "results" / "feature_comparison_2015_2024"))
    args = parser.parse_args()
    output = ROOT / "results" / "augmented_price_2015_2024"
    output.mkdir(parents=True, exist_ok=args.resume)
    started = time.monotonic()
    data_path = ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv"
    with open(ROOT / "results" / "final_config.json") as handle:
        selection = json.load(handle)
    with open(args.reference_results / "config.json") as handle:
        assert selection == json.load(handle)
    config = selection["price"]["config"]
    source_paths = [Path(__file__), ROOT / "code" / "price_regression.py",
                    ROOT / "code" / "forecast_package" / "model.py",
                    ROOT / "code" / "forecast_package" / "train.py",
                    ROOT / "code" / "run_experiments.py"]
    fingerprint = {"config": config, "data_hash": digest(data_path),
                   "sources": {str(p.relative_to(ROOT)): digest(p) for p in source_paths}}
    metadata_path = output / "frozen_config.json"
    if metadata_path.exists():
        assert json.loads(metadata_path.read_text()) == fingerprint, "Resume inputs changed"
    else:
        metadata_path.write_text(json.dumps(fingerprint, indent=2))

    frame = load_data(data_path).loc[:"2025-01-01 00:00:00"]
    features, labels = training_table(frame)
    assert features.shape[1] == 571
    feature_days = market_days(features.index)
    means = frame["nsw_price"].groupby(market_days(frame.index)).mean().shift(1)
    parts, regression_audit = [], []
    for year in (2011, 2012, 2013, 2014):
        path = (ROOT / "results" / "price_stacking_2014" / f"oof_{year}.csv"
                if year < 2014 else ROOT / "results" / "price_regression_2014" / "predictions.csv")
        part = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        assert (market_days(part.index).year == year).all()
        pd.testing.assert_series_equal(labels.loc[part.index, "price"], part.actual_up,
                                       check_names=False, check_dtype=False)
        parts.append(part[["nsw_price_pred", "nsw_price_margin"]])
        regression_audit.append({"forecast_start": f"{year}-01-01", "forecast_end": f"{year}-12-31",
                                 "training_end": f"{year-1}-12-31", "source": str(path)})

    # Use the same biennial update schedule as the deployed candidate.
    for start, end in ((2015, 2015), (2016, 2017), (2018, 2019),
                       (2020, 2021), (2022, 2023), (2024, 2024)):
        train = feature_days < pd.Timestamp(f"{start}-01-01")
        test = (feature_days.year >= start) & (feature_days.year <= end)
        assert feature_days[train].max() < feature_days[test].min()
        path = output / f"regression_{start}_{end}.csv"
        print(f"Regression features for {start}–{end}", flush=True)
        if args.resume and path.exists():
            part = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        else:
            regressor = fit_price_regressor(features.loc[train],
                                            frame.loc[features.index[train], "nsw_price"], config)
            threshold = pd.Series(feature_days[test], index=features.index[test]).map(means)
            part = predict_price_regressor(regressor, features.loc[test], threshold)
            part.to_csv(path, index_label="timestamp")
        assert part.index.equals(features.index[test])
        parts.append(part[["nsw_price_pred", "nsw_price_margin"]])
        regression_audit.append({"forecast_start": f"{start}-01-01", "forecast_end": f"{end}-12-31",
                                 "training_end": str(feature_days[train].max().date()),
                                 "source": str(path)})
        pd.DataFrame(regression_audit).to_csv(output / "regression_audit.csv", index=False)
        print(f"Saved regression features for {start}–{end}", flush=True)

    regression = pd.concat(parts).sort_index().rename(columns={
        "nsw_price_pred": "regressed_price", "nsw_price_margin": "regressed_price_margin"})
    assert regression.index.is_unique
    matched = features.loc[feature_days.year >= 2011]
    assert matched.index.equals(regression.index)
    augmented = matched.join(regression)
    assert augmented.shape[1] == 573 and np.isfinite(augmented.to_numpy()).all()
    days_all = market_days(matched.index)
    evaluation = days_all >= pd.Timestamp("2015-01-01")
    days = days_all[evaluation].unique()
    assert days[0] == pd.Timestamp("2015-01-01") and days[-1] == pd.Timestamp("2024-12-31")
    truth = labels.loc[matched.index[evaluation], "price"].to_numpy().reshape(-1, 48)
    np.savez(output / "truth.npz", days=days.to_numpy(), price=truth)
    forecasts = {}
    for name, table in (("matched_reference", matched), ("augmented", augmented)):
        path = output / f"{name}.npy"
        print(f"Starting classifier simulation: {name}", flush=True)
        if args.resume and path.exists():
            forecasts[name] = np.load(path)
        else:
            initial = days_all.year <= 2014
            model = fit_model("price", config, table.loc[initial], labels.loc[table.index[initial]])
            forecasts[name], retrains, _ = simulate_policy(
                "price", "periodic_expanding", model, config, table, labels.loc[table.index],
                days, pd.Series(0.0, index=days), np.zeros(365), np.zeros(365))
            assert len(retrains) == 5
            np.save(path, forecasts[name])
            pd.DataFrame(retrains).to_csv(output / f"{name}_retrains.csv", index=False)
        assert forecasts[name].shape == truth.shape
        partial = dict(model=name, **price_metrics(truth.ravel(), forecasts[name].ravel()))
        (output / f"{name}_metrics.json").write_text(json.dumps(partial, indent=2))
        print(partial, flush=True)

    forecasts["current_classifier_2010_onwards"] = np.load(args.reference_results / "current_price.npy")
    totals, periods = [], []
    for name, prediction in forecasts.items():
        assert prediction.shape == truth.shape
        totals.append(dict(model=name, **price_metrics(truth.ravel(), prediction.ravel())))
        for frequency in ("M", "Y"):
            period_index = days.to_period(frequency)
            for period in period_index.unique():
                mask = period_index == period
                row = dict(model=name, frequency=frequency, period=str(period),
                           **price_metrics(truth[mask].ravel(), prediction[mask].ravel()))
                if frequency == "M":
                    daily_correct = (truth[mask] == prediction[mask]).mean(axis=1)
                    rng = np.random.default_rng(9417)
                    draws = rng.integers(0, len(daily_correct), size=(1000, len(daily_correct)))
                    lo, hi = np.percentile(daily_correct[draws].mean(axis=1)*100, [2.5, 97.5])
                    row.update(accuracy_day_boot_low=float(lo), accuracy_day_boot_high=float(hi))
                periods.append(row)
    pd.DataFrame(totals).to_csv(output / "aggregate_metrics.csv", index=False)
    pd.DataFrame(periods).to_csv(output / "period_metrics.csv", index=False)
    (output / "completion.json").write_text(json.dumps({"status": "passed",
        "seconds": time.monotonic()-started, "holdout_evaluated": False}, indent=2))
    print(pd.DataFrame(totals).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
