"""Pure test (not in the repo): explicit negative-price features for every harness row (target slot t of day D+1),
built only from prices of day D and earlier (known at the end of day D). Saved as runs/negprice.npy (+ names json).

  nsw_neg_slots_D          count of day-D NSW slots with price < 0
  nsw_neg_slots_wk         count on day D-6 (same weekday as the target day)
  nsw_neg_slots_7d / _28d  totals over days D-6..D / D-27..D
  nsw_neg_slot_lag1        1 if this slot was negative on day D
  nsw_neg_slot_share_7d    share of days D-6..D on which this slot was negative
  nsw_negmass_D_slog       signed log of the sum of negative NSW prices on day D (<= 0)
  vic/qld_neg_slots_D, vic/qld_neg_slots_7d
Names avoid the substring 'price' so the neural net does not signed-log the counts."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import harness
from model import market_days


def build(raw, index):
    """raw: DataFrame indexed by timestamp with nsw/vic/qld_price; index: harness row timestamps (day D+1 slots)."""
    rd = market_days(raw.index)
    slot = raw.groupby(rd).cumcount().to_numpy()
    out_days = market_days(index)
    prev = out_days - pd.Timedelta(days=1)                       # day D for each row
    row_slot = pd.Series(1, index=index).groupby(np.asarray(out_days)).cumcount().to_numpy()
    feats = {}
    for st in ("nsw", "vic", "qld"):
        neg = (raw[f"{st}_price"] < 0).astype(float)
        per_day = neg.groupby(rd).sum()                          # negative slots per market day
        full = per_day.reindex(pd.date_range(per_day.index.min(), per_day.index.max()), fill_value=0.0)
        feats[f"{st}_neg_slots_D"] = full.reindex(prev).to_numpy()
        feats[f"{st}_neg_slots_7d"] = full.rolling(7, min_periods=1).sum().reindex(prev).to_numpy()
        if st == "nsw":
            feats["nsw_neg_slots_wk"] = full.reindex(prev - pd.Timedelta(days=6)).to_numpy()
            feats["nsw_neg_slots_28d"] = full.rolling(28, min_periods=1).sum().reindex(prev).to_numpy()
            # slot-level: was this slot negative on day D; share of D-6..D with this slot negative
            grid = pd.DataFrame({"d": rd, "s": slot, "v": neg.to_numpy()}).pivot_table(index="d", columns="s", values="v", aggfunc="first")
            grid = grid.reindex(pd.date_range(grid.index.min(), grid.index.max()))
            roll = grid.rolling(7, min_periods=1).mean()
            r = grid.index.get_indexer(prev); s = np.clip(row_slot, 0, grid.shape[1] - 1)
            feats["nsw_neg_slot_lag1"] = np.where(r >= 0, grid.to_numpy()[np.clip(r, 0, None), s], np.nan)
            feats["nsw_neg_slot_share_7d"] = np.where(r >= 0, roll.to_numpy()[np.clip(r, 0, None), s], np.nan)
            mass = raw["nsw_price"].clip(upper=0).groupby(rd).sum()
            feats["nsw_negmass_D_slog"] = (-np.log1p(-mass)).reindex(prev).to_numpy()
    return pd.DataFrame(feats, index=index)


if __name__ == "__main__":
    raw = pd.read_csv(f"{harness.E}/data_upto_2024.csv", usecols=["timestamp", "nsw_price", "vic_price", "qld_price"],
                      parse_dates=["timestamp"]).set_index("timestamp")
    assert raw.index.max() == pd.Timestamp("2025-01-01 00:00")
    X, truth, days = harness.load()
    F = build(raw, X.index)
    # look-ahead check: changing every day-D+1 price (the target day) must not change that day's features
    target_day = days.max()
    raw2 = raw.copy(); raw2.loc[market_days(raw2.index) == target_day, ["nsw_price", "vic_price", "qld_price"]] = -999.0
    F2 = build(raw2, X.index)
    m = np.asarray(days == target_day)
    assert np.allclose(F.to_numpy()[m], F2.to_numpy()[m], equal_nan=True), "features depend on the target day!"
    print("look-ahead check passed: target-day prices do not affect that day's features")
    yrs = np.asarray(days.year)
    print("rows with any NaN by year (2010 start expected):", {int(y): int(F[yrs == y].isna().any(axis=1).sum()) for y in (2010, 2011, 2024)})
    d1 = pd.Series(F["nsw_neg_slots_D"].to_numpy(), index=np.asarray(days)).groupby(level=0).first()
    print("target days whose day D had >= 6 NSW negative slots, by year:", {int(y): int(((d1 >= 6) & (d1.index.year == y)).sum()) for y in range(2014, 2025)})
    np.save(f"{harness.RUNS}/negprice.npy", F.to_numpy(np.float32))
    json.dump(list(F.columns), open(f"{harness.RUNS}/negprice_names.json", "w"))
    print("saved", F.shape, list(F.columns))
