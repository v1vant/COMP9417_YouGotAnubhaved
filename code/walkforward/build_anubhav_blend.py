"""Pure test (not in the repo): rebuild Anubhav's day-ahead demand forecast (0.5 LightGBM + 0.5 MLP), aligned to
the harness rows, and save it as runs/anubhav_blend_forecast.npy.
- 2011-2014: his yearly out-of-sample folds (commit 4f32442, results/forecast_demand_price_2014/demand_forecasts_Y.csv).
- 2015-2024: 0.5 * (results/feature_comparison_2015_2024/current_demand.npy + results/demand_mlp_10epoch_2015_2024),
  both retrained on the team schedule (training ends 2014, 2015, 2017, 2019, 2021, 2023).
Only files covering 2011-2024 are read."""
import io, os, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
REPO = harness.REPO
from model import market_days

index = pd.DatetimeIndex(np.load(f"{harness.E}/index.npy", allow_pickle=True))
days = market_days(index)
assert days.max() == pd.Timestamp("2024-12-31")
truth = pd.read_parquet(f"{harness.E}/truth.parquet")["demand"].to_numpy(np.float64)

early = []
for year in range(2011, 2015):
    raw = subprocess.run(["git", "-C", REPO, "show", f"4f32442:results/forecast_demand_price_2014/demand_forecasts_{year}.csv"],
                         capture_output=True, text=True, check=True).stdout
    d = pd.read_csv(io.StringIO(raw), index_col="timestamp", parse_dates=True)
    assert np.allclose(d.forecast, 0.5 * (d.lgbm + d.mlp))
    assert (market_days(d.index).year == year).all()
    early.append(d.forecast)
early = pd.concat(early)

mlp = pd.concat([pd.read_csv(f"{REPO}/results/demand_mlp_10epoch_2015_2024/predictions_{y}.csv",
                             index_col="timestamp", parse_dates=True).predicted for y in range(2015, 2025)])
lgbm = np.load(f"{REPO}/results/feature_comparison_2015_2024/current_demand.npy").reshape(-1)
assert len(lgbm) == len(mlp) == 175344 and market_days(mlp.index).max() == pd.Timestamp("2024-12-31")
late = pd.Series(0.5 * (lgbm + mlp.to_numpy()), index=mlp.index)

series = pd.concat([early, late])
assert series.index.is_unique and series.index.is_monotonic_increasing
f = series.reindex(index).to_numpy(np.float64)
keep = np.asarray(days.year >= 2011)
assert np.isfinite(f[keep]).all() and np.isnan(f[~keep]).all()

try:   # optional: compare with our LightGBM-only schedule-matched forecast if those runs exist
    ours = harness.demand_forecast(days, True).astype(np.float64)   # our LightGBM-only, schedule-matched
    print("Yearly MAPE % (actual demand): Anubhav 50/50 blend vs our LightGBM-only schedule-matched forecast")
    for y in range(2011, 2025):
        m = np.asarray(days.year == y)
        a = 100 * np.mean(np.abs(f[m] - truth[m]) / truth[m]); b = 100 * np.mean(np.abs(ours[m] - truth[m]) / truth[m])
        print(f"  {y}: blend {a:.3f}  ours {b:.3f}  ({a - b:+.3f})")
    m = np.asarray(days.year >= 2015)
    print(f"  2015-2024 pooled: blend {100*np.mean(np.abs(f[m]-truth[m])/truth[m]):.3f}  ours {100*np.mean(np.abs(ours[m]-truth[m])/truth[m]):.3f}")
except FileNotFoundError:
    print('(our LightGBM-only demand forecasts not present; comparison skipped)')
np.save(f"{harness.RUNS}/anubhav_blend_forecast.npy", f)
print("saved", f"{harness.RUNS}/anubhav_blend_forecast.npy", f.shape)
