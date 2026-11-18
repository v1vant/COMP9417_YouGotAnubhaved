"""Pure test (not in the repo): Suhan's 26 weather-shape inputs (Sydney temperature/solar shape of the forecast
day, from the supplied next-day weather only), aligned to the harness rows -> runs/weather_shape.npy.
Built with our package's feature_frame(weather_shape=True) on data_upto_2024.csv (ends 2025-01-01 00:00)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import harness
from model import load_data, training_table

NAMES = []
for col in ("syd_temp", "syd_solar"):          # recipe.weather_shape_names() at Suhan's ad82bc5, same order
    for offset in (2, 4):
        NAMES += [f"{col}_before_{offset}", f"{col}_after_{offset}", f"{col}_ramp_{offset}", f"{col}_ramp_{offset}_change_1d"]
    NAMES += [f"{col}_daily_{stat}_change_1d" for stat in ("mean", "min", "max")]
    NAMES += [f"{col}_peak_slot", f"{col}_peak_slot_change_1d"]
X, truth, days = harness.load()
feats, _ = training_table(load_data(f"{harness.E}/data_upto_2024.csv"), weather_shape=True)
assert set(NAMES) <= set(feats.columns), sorted(set(NAMES) - set(feats.columns))
extra = feats.reindex(X.index)
assert extra.index.equals(X.index) and np.isfinite(extra[NAMES].to_numpy()).all()
base_cols = [c for c in X.columns]
diff = np.nanmax(np.abs(extra[base_cols].to_numpy(np.float32) - X.to_numpy(np.float32)))
print(f"{len(NAMES)} weather-shape columns; base 571 columns identical to the cache: max |diff| {diff}")
np.save(f"{harness.RUNS}/weather_shape.npy", extra[NAMES].to_numpy(np.float32))
json.dump(NAMES, open(f"{harness.RUNS}/weather_shape_names.json", "w"))
print("saved", extra[NAMES].shape)
