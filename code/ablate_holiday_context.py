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
from model import holiday_context_features, load_data, market_days, predict_task, training_table
from train import SEED, fit_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    parser.add_argument("--selection", type=Path, default=ROOT / "results/final_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/holiday_ablation")
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
    for name, enabled in (("baseline", False), ("holiday_context", True)):
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            train = days < begin
            valid = (days >= begin) & (days < end)
            print(f"{name}: fitting through {(begin - pd.Timedelta(days=1)).date()}, "
                  f"forecasting {begin.year}-{end.year - 1}", flush=True)
            model = fit_model("demand", dict(config, holiday_context=enabled),
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
    adjacent = (context.holiday_yesterday > 0) | (context.holiday_tomorrow > 0)
    holiday = context.holiday > 0
    shutdown = context.christmas_shutdown > 0
    groups = {"all": np.ones(len(data), dtype=bool), "holiday": holiday,
              "adjacent_nonholiday": adjacent & ~holiday,
              "christmas_shutdown": shutdown,
              "ordinary": ~(holiday | adjacent | shutdown),
              "midday": context.slot.isin([23, 24, 25, 26]),
              "evening": context.slot.isin([37, 38, 39])}
    rows, daily_rows = [], []
    for name in ("baseline", "holiday_context"):
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
            daily = ape.loc[mask].groupby(data.loc[mask, "day"]).mean()
            daily_rows.extend({"variant": name, "group": group, "day": day, "MAPE": value}
                              for day, value in daily.items())
    metrics = pd.DataFrame(rows)
    daily = pd.DataFrame(daily_rows)
    paired = daily.pivot(index=["group", "day"], columns="variant", values="MAPE")
    paired["delta_MAPE_pp"] = paired.holiday_context - paired.baseline
    # Resample paired days, not individual half-hours. Negative delta is better.
    uncertainty = []
    for group, values in paired.groupby(level="group"):
        delta = values.delta_MAPE_pp.to_numpy()
        rng = np.random.default_rng(SEED)
        draws = rng.integers(len(delta), size=(2000, len(delta)))
        low, high = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
        uncertainty.append({"group": group, "delta_MAPE_pp": delta.mean(),
                            "day_boot_low": low, "day_boot_high": high})
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    paired.to_csv(args.output / "daily_paired_errors.csv")
    pd.DataFrame(uncertainty).to_csv(args.output / "paired_deltas.csv", index=False)
    pd.DataFrame(fits).to_csv(args.output / "fits.csv", index=False)
    sources = [Path(__file__), HERE / "forecast_package/model.py", HERE / "forecast_package/train.py"]
    manifest = {"seed": SEED, "config": config, "quick": args.quick,
                "evaluation": "2015-01-01 through 2024-12-31 (development)",
                "extra_features": ["holiday_yesterday", "holiday_tomorrow", "christmas_shutdown"],
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
