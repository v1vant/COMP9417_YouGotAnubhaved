"""Rebuild artifacts/ from the released CSV only: python train.py path/to/nem_weather_halfhourly_2010_2026.csv

1. reference_train.py rebuilds the unchanged reference demand blend (and a
   reference price model that is replaced below).
2. The price bundle is refitted from candidate.json on the harness-exact strict
   target: 571 features (+26 weather-shape inputs if selected), one LightGBM
   member per seed via recipe.fit_price, fixed cutoff. Fixed seeds, no network, CPU only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recipe import abs_margin, fit_price, weather_shape_names  # noqa: E402

FIELDS = {'name', 'seeds', 'half_life_years', 'threshold', 'source', 'weather_shape', 'params', 'near_threshold_scale'}


def price_table(frame, weather_shape=False):
    """Rows/labels of code/price_v2/pipeline.py:table; columns = 571 base (+ weather shape, fixed order)."""
    from model import market_days, training_table
    x, y = training_table(frame, weather_shape=weather_shape)
    if weather_shape:
        shape = weather_shape_names()
        x = x.loc[:, [c for c in x.columns if c not in shape] + shape]
    means = frame.nsw_price.groupby(market_days(frame.index)).agg(lambda s: s.mean()).shift(1)
    y['price'] = (frame.loc[x.index, 'nsw_price'].to_numpy() >
                  means.reindex(market_days(x.index)).to_numpy()).astype(int)
    counts = pd.Series(1, index=x.index).groupby(market_days(x.index)).sum()
    keep = market_days(x.index).isin(counts[counts == 48].index)
    return x.loc[keep], y.loc[keep]


def fit_members(frame, candidate):
    unknown = set(candidate) - FIELDS
    if unknown or not candidate.get('seeds'):
        raise ValueError(f'Unsupported candidate fields: {sorted(unknown)}')
    shape = bool(candidate.get('weather_shape'))
    x, y = price_table(frame, shape)
    expected = 571 + (26 if shape else 0)
    if x.shape[1] != expected:
        raise ValueError(f'Expected {expected} price features, got {x.shape[1]}')
    margin = abs_margin(frame).loc[x.index].to_numpy() if candidate.get('near_threshold_scale') else None
    members = [fit_price(x, y.price, seed, candidate.get('params'), candidate.get('half_life_years'),
                         candidate.get('near_threshold_scale'), margin) for seed in candidate['seeds']]
    return x, members


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv', type=Path)
    args = ap.parse_args()
    started = time.monotonic()
    candidate = json.loads((HERE/'candidate.json').read_text())
    subprocess.run([sys.executable, str(HERE/'reference_train.py'), str(args.csv.resolve())], check=True)
    demand_seconds = time.monotonic() - started
    from threadpoolctl import threadpool_limits
    from model import load_data, market_days
    with threadpool_limits(limits=2):
        frame = load_data(args.csv)
        x, members = fit_members(frame, candidate)
    days = market_days(x.index)
    path = HERE/'artifacts'
    models = joblib.load(path/'models.joblib')
    models['price'] = dict(task='price', config=candidate, features=list(x.columns), estimator=members[0],
                           members=members, threshold=float(candidate['threshold']),
                           training_start=str(days.min()), training_end=str(days.max()))
    joblib.dump(models, path/'models.joblib', compress=3)
    manifest = json.loads((path/'manifest.json').read_text())
    manifest['price_v3'] = dict(candidate=candidate, training_end=str(days.max()), training_rows=len(x),
                                features=x.shape[1])
    manifest['rebuild_seconds'] = time.monotonic() - started
    manifest['demand_rebuild_seconds'] = demand_seconds
    (path/'manifest.json').write_text(json.dumps(manifest, indent=2))
    size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    if size > 100_000_000:
        raise ValueError(f'Artifacts exceed 100 MB: {size}')
    print(f"Rebuilt artifacts in {manifest['rebuild_seconds']:.0f}s ({size/1e6:.1f} MB)", flush=True)


if __name__ == '__main__':
    main()
