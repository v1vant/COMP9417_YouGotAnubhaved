"""Paired demand ablation: fixed LightGBM settings, 2015-2024 development only.

Initial fit ends in 2014; expanding-history replacements start on 1 January
2016/2018/2020/2022/2024. Both variants are refitted with identical seeds and
parameters. This is a development experiment prompted by inspected errors,
not independent validation. No final artifacts or previous results are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "forecast_package"))
from model import (holiday_context_features, load_data, market_days, predict_task,
                   solar_trend_features, training_table)
from train import SEED, fit_model


def save_overview(data, names, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)
    for name in names:
        error = data[name] - data.actual
        values = pd.DataFrame({"MAPE": 100 * error.abs() / data.actual,
                               "MAE": error.abs(), "bias": error}, index=data.index)
        slots = values.groupby(data.slot).mean()
        for ax, column in zip(axes[:3], ("MAPE", "MAE", "bias")):
            ax.plot(slots.index / 2, slots[column], label=name)
        monthly = values.MAPE.groupby(data.day.dt.to_period("M")).mean()
        axes[3].plot(monthly.index.to_timestamp(), monthly, label=name)
    for ax, label in zip(axes, ("MAPE (%)", "MAE (MW)", "Prediction − actual (MW)", "MAPE (%)")):
        ax.set_ylabel(label)
        ax.legend()
    for ax in axes[:3]:
        ax.axvspan(10, 15, alpha=0.1, color="grey")
        ax.set_xlabel("Interval start (AEST hour)")
    axes[2].axhline(0, color="grey", linestyle="--")
    axes[3].set_xlabel("Month")
    fig.suptitle("Paired demand ablation: 2015–2024 development")
    fig.savefig(output / "error_overview.png", dpi=160)
    plt.close(fig)


def main(feature_set="holiday"):
    if feature_set not in ("holiday", "solar_trend"):
        raise ValueError("Unknown ablation feature set")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    parser.add_argument("--selection", type=Path, default=ROOT / "results/final_config.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / ("holiday_ablation" if feature_set == "holiday"
                                                   else "solar_trend_ablation"))
    parser.add_argument("--quick", action="store_true", help="120-tree smoke check only")
    args = parser.parse_args()
    started = time.monotonic()
    with args.selection.open() as handle:
        policy = json.load(handle)["demand"]
    config = dict(policy["config"])
    if (config["family"] != "boosted" or "half_life_years" in config
            or policy["strategy"] != "periodic_expanding" or policy.get("window_years")):
        parser.error("This ablation requires standard LightGBM with periodic expanding history")
    if args.quick:
        config["trees"] = min(config["trees"], 120)
    variants = (("baseline", {"holiday_context": False, "solar_trend": False}),
                ("holiday_context", {"holiday_context": True, "solar_trend": False}))
    if feature_set == "solar_trend":
        variants = (("holiday_context", {"holiday_context": True, "solar_trend": False}),
                    ("holiday_solar_trend", {"holiday_context": True, "solar_trend": True}))
    reference, candidate = [name for name, _ in variants]
    frame = load_data(args.data)
    frame = frame.loc[market_days(frame.index) <= pd.Timestamp("2024-12-31")]
    if market_days(frame.index).max() != pd.Timestamp("2024-12-31"):
        parser.error("Data must include complete development data through 2024")
    x, y = training_table(frame)
    days = market_days(x.index)
    evaluation = days >= pd.Timestamp("2015-01-01")
    index = x.index[evaluation]
    data = pd.DataFrame({"actual": y.loc[evaluation, "demand"]}, index=index)
    fits = []
    boundaries = pd.to_datetime(["2015-01-01", "2016-01-01", "2018-01-01",
                                 "2020-01-01", "2022-01-01", "2024-01-01", "2025-01-01"])
    for name, options in variants:
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            train = days < begin
            valid = (days >= begin) & (days < end)
            print(f"{name}: fitting through {(begin - pd.Timedelta(days=1)).date()}, "
                  f"forecasting {begin.year}-{end.year - 1}", flush=True)
            model = fit_model("demand", dict(config, **options),
                              x.loc[train], y.loc[train])
            assert pd.Timestamp(model["training_end"]) < begin
            data.loc[x.index[valid], name] = predict_task(model, x.loc[valid])
            fits.append({"variant": name, "effective_day": str(begin.date()),
                         "training_end": model["training_end"], "training_rows": int(train.sum())})
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Incomplete ablation predictions")
    context = holiday_context_features(x.loc[evaluation])
    data["day"] = market_days(data.index)
    data["year"] = data.day.dt.year
    data["slot"] = x.loc[evaluation, "slot"]
    data["solar_delta_anchor"] = solar_trend_features(x.loc[evaluation])["solar_delta_anchor"]
    adjacent = (context.holiday_yesterday > 0) | (context.holiday_tomorrow > 0)
    holiday = context.holiday > 0
    shutdown = context.christmas_shutdown > 0
    groups = {"all": np.ones(len(data), dtype=bool), "holiday": holiday,
              "adjacent_nonholiday": adjacent & ~holiday,
              "christmas_shutdown": shutdown,
              "ordinary": ~(holiday | adjacent | shutdown),
              "midday": context.slot.isin([23, 24, 25, 26]),
              "evening": context.slot.isin([37, 38, 39])}
    if feature_set == "solar_trend":
        daytime = (context.slot >= 20) & (context.slot < 30)
        groups["midday_10_15"] = daytime
        groups["other_hours"] = ~daytime
        solar_bins = pd.cut(data.solar_delta_anchor, [-np.inf, -200, -50, 50, 200, np.inf],
                            labels=["large_decrease", "decrease", "similar", "increase", "large_increase"])
        for label in solar_bins.cat.categories:
            groups[f"midday_solar_{label}"] = daytime & (solar_bins == label)
    rows, daily_rows = [], []
    for name, _ in variants:
        error = data[name] - data.actual
        ape = 100 * error.abs() / data.actual
        for group, mask in groups.items():
            for year in ["all"] + sorted(data.year.unique().tolist()):
                selected = mask & (True if year == "all" else data.year == year)
                rows.append({"variant": name, "group": group, "year": year,
                             "intervals": int(selected.sum()),
                             "MAPE": float(ape.loc[selected].mean()),
                             "MAE": float(error.loc[selected].abs().mean()),
                             "bias_mw": float(error.loc[selected].mean())})
            daily = ape.loc[mask].groupby(data.loc[mask, "day"]).agg(["mean", "size"])
            daily_rows.extend({"variant": name, "group": group, "day": day,
                               "MAPE": row["mean"], "intervals": int(row["size"])}
                              for day, row in daily.iterrows())
    metrics = pd.DataFrame(rows)
    daily = pd.DataFrame(daily_rows)
    paired = daily.pivot(index=["group", "day"], columns="variant", values="MAPE")
    paired["delta_MAPE_pp"] = paired[candidate] - paired[reference]
    paired["intervals"] = daily.loc[daily.variant == reference].set_index(["group", "day"]).intervals
    # Resample paired days, not individual half-hours. Negative delta is better.
    uncertainty = []
    for group, values in paired.groupby(level="group"):
        delta = values.delta_MAPE_pp.to_numpy()
        counts = values.intervals.to_numpy()
        rng = np.random.default_rng(SEED)
        draws = rng.integers(len(delta), size=(2000, len(delta)))
        boot = (delta * counts)[draws].sum(axis=1) / counts[draws].sum(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        uncertainty.append({"group": group, "delta_MAPE_pp": np.average(delta, weights=counts),
                            "day_boot_low": low, "day_boot_high": high})
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    yearly = metrics.pivot(index=["group", "year"], columns="variant", values="MAPE")
    yearly["delta_MAPE_pp"] = yearly[candidate] - yearly[reference]
    yearly.to_csv(args.output / "yearly_deltas.csv")
    paired.to_csv(args.output / "daily_paired_errors.csv")
    pd.DataFrame(uncertainty).to_csv(args.output / "paired_deltas.csv", index=False)
    pd.DataFrame(fits).to_csv(args.output / "fits.csv", index=False)
    data.to_parquet(args.output / "half_hour_predictions.parquet")
    save_overview(data, (reference, candidate), args.output)
    sources = [Path(__file__), HERE / "forecast_package/model.py", HERE / "forecast_package/train.py"]
    if feature_set == "solar_trend":
        sources.append(HERE / "ablate_solar_trend.py")
    manifest = {"seed": SEED, "config": config, "quick": args.quick,
                "evaluation": "2015-01-01 through 2024-12-31 (development)",
                "feature_set": feature_set, "variants": dict(variants),
                "extra_features": (["holiday_yesterday", "holiday_tomorrow", "christmas_shutdown"]
                                   if feature_set == "holiday" else
                                   ["solar_delta_anchor", "trend_years", "solar_delta_trend"]),
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources},
                "data_hash_through_2024": hashlib.sha256(
                    pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest(),
                "seconds": time.monotonic() - started,
                "note": "Paired day bootstrap; exploratory intervals do not adjust for experiment selection"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(metrics.loc[metrics.year == "all"].to_string(index=False), flush=True)
    print(pd.DataFrame(uncertainty).to_string(index=False), flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
