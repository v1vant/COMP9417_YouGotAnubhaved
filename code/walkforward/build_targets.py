"""Pure test (not in the repo): continuous targets for every harness row (target interval t of market day D+1).
price  = actual NSW price at t
thr    = mean NSW price of market day D (the label's threshold)
margin = price - thr                      (label = margin > 0)
mean_next = mean NSW price of market day D+1 (the target day), so margin = level + deviation with
level = mean_next - thr (one value per day) and deviation = price - mean_next (intraday).
Reads only data_upto_2024.csv (ends 2025-01-01 00:00 = last interval of market day 2024-12-31)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import harness
from model import market_days

raw = pd.read_csv(f"{harness.E}/data_upto_2024.csv", usecols=["timestamp", "nsw_price"], parse_dates=["timestamp"]).set_index("timestamp")
assert raw.index.max() == pd.Timestamp("2025-01-01 00:00")
raw_days = market_days(raw.index)
counts = raw.groupby(raw_days).size()
day_mean = raw["nsw_price"].groupby(raw_days).mean()

X, truth, days = harness.load()
idx = X.index
price = raw["nsw_price"].reindex(idx).to_numpy(np.float64)
thr = day_mean.reindex(days - pd.Timedelta(days=1)).to_numpy(np.float64)
mean_next = day_mean.reindex(days).to_numpy(np.float64)
assert np.isfinite(price).all() and np.isfinite(thr).all() and np.isfinite(mean_next).all()
assert (counts.reindex(days).to_numpy() == 48).all(), "every target day must be complete"
margin = price - thr
label = truth["price"].to_numpy().astype(bool)
mismatch = int(((margin > 0) != label).sum())
print(f"rows {len(idx)}; label == (margin > 0) for all but {mismatch} rows")
print("threshold vs feature price_threshold, max |diff|:", float(np.max(np.abs(thr - X["price_threshold"].to_numpy(np.float64)))))
np.savez(f"{harness.RUNS}/targets.npz", price=price, thr=thr, margin=margin, mean_next=mean_next)
q = np.percentile(margin, [0.1, 1, 25, 50, 75, 99, 99.9])
print("margin percentiles 0.1/1/25/50/75/99/99.9:", np.round(q, 1))
n = pd.Series(label.astype(int)).groupby(np.asarray(days)).sum()
print("up slots per day: mean", round(n.mean(), 2), "std", round(n.std(), 2), "share of days with 0 or 48:", round(float(((n == 0) | (n == 48)).mean()), 3))
