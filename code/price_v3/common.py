"""Shared helpers for v3 price research. Imports v2 code unchanged (its hashes are recorded)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / 'code' / 'price_v2'
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))
DATA = ROOT / 'data' / 'nem_weather_halfhourly_2010_2026.csv'
THRESHOLD = .57


def market_days(index):
    return (index - pd.Timedelta(minutes=30)).normalize()


def balanced_accuracy(y, p):
    """Harness definition: mean recall over classes present."""
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    return float(100 * np.mean([np.mean(p[y == k] == k) for k in np.unique(y)]))


def oracle(y, probability):
    """Best-in-hindsight BA over the research cutoff grid. Diagnostic only, never for selection."""
    grid = np.round(np.arange(.05, .951, .01), 2)
    scores = [balanced_accuracy(y, probability > t) for t in grid]
    i = int(np.argmax(scores))
    return scores[i], float(grid[i])


def harness_targets(frame):
    """Strict target and both baselines, using the harness-exact Series.mean()."""
    days = market_days(frame.index)
    price = frame.nsw_price
    means = price.groupby(days).agg(lambda s: s.mean())
    tau = pd.Series(days, index=frame.index).map(means.shift(1))
    return pd.DataFrame(dict(
        actual=(price > tau).astype(int),
        yesterday=(price.shift(48) > tau).astype(int),
        last_week=(price.shift(336) > tau).astype(int),
        tau=tau, price=price), index=frame.index)


def weekly_gain(actual, prediction, baseline):
    """BA gain over a baseline for complete Monday-Sunday market weeks (the live-week shape)."""
    frame = pd.DataFrame(dict(a=np.asarray(actual), p=np.asarray(prediction), b=np.asarray(baseline)),
                         index=actual.index)
    week = market_days(frame.index).to_period('W-SUN')
    rows = []
    for label, g in frame.groupby(week):
        if len(g) != 7 * 48:
            continue
        model, base = balanced_accuracy(g.a, g.p), balanced_accuracy(g.a, g.b)
        rows.append(dict(week=str(label), start=label.start_time.date(), BA=model, baseline_BA=base, G=model - base))
    return pd.DataFrame(rows)
