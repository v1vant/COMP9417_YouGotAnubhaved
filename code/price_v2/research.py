"""Price-only research. Never scores post-2024 outcomes or replaces demand artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import load_data, market_days, training_table, predict_price_probability
from price_context import extend_price_features
from price_train import fit_model, SEED

BASE = dict(family='boosted', leaves=31, trees=600, learning_rate=.04, threshold=.57)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(recall_0=float(100*np.mean(p[y==0]==0)) if np.any(y==0) else None,
                recall_1=float(100*np.mean(p[y==1]==1)) if np.any(y==1) else None,
                accuracy=float(100*np.mean(y == p)), balanced_accuracy=float(100*np.mean(
        [np.mean(p[y == k] == k) for k in np.unique(y)])))


def threshold(y, p, previous=.57):
    grid = np.round(np.arange(.05, .951, .01), 2)
    scores = np.array([metrics(y, p > t)['balanced_accuracy'] for t in grid])
    ties = grid[np.isclose(scores, scores.max(), rtol=0, atol=1e-10)]
    return float(ties[np.argmin(abs(ties-previous))])


def fit(task_config, x, y, cutoff, window=0):
    days = market_days(x.index)
    mask = days < cutoff
    if window:
        mask &= days >= cutoff - pd.DateOffset(years=window)
    return fit_model('price', task_config, x.loc[mask], y.loc[mask])


def refresh(config, x, y, cutoff, window, previous):
    start = cutoff - pd.Timedelta(days=365)
    middle = start + pd.Timedelta(days=182)
    days = market_days(x.index)
    probs, labels = [], []
    for a, b in [(start, middle), (middle, cutoff)]:
        m = fit(config, x, y, a, window)
        mask = (days >= a) & (days < b)
        probs.extend(predict_price_probability(m, x.loc[mask]))
        labels.extend(y.loc[mask, 'price'])
    return threshold(labels, np.array(probs), previous)


def backtest(config, x, y, end, window=0, adaptive=False):
    days = market_days(x.index)
    boundaries = list(pd.to_datetime(['2015-01-01','2016-01-01','2018-01-01',
                                      '2020-01-01','2022-01-01','2024-01-01']))
    rows, fits = [], []
    frozen = None
    t = config.get('threshold', .57)
    for start, stop in zip(boundaries, boundaries[1:] + [end + pd.Timedelta(days=1)]):
        if start > end:
            break
        if adaptive:
            t = refresh(config, x, y, start, window, t)
        began = time.monotonic()
        m = fit(config, x, y, start, window)
        if frozen is None:
            frozen = m
        mask = (days >= start) & (days < min(stop, end + pd.Timedelta(days=1)))
        p = predict_price_probability(m, x.loc[mask])
        fp = predict_price_probability(frozen, x.loc[mask])
        rows.append(pd.DataFrame(dict(actual=y.loc[mask, 'price'], probability=p,
                                     frozen_prediction=(fp > config.get('threshold', .57)).astype(int),
                                     prediction=(p > t).astype(int), threshold=t)))
        fits.append(dict(effective_day=str(start.date()), training_start=m['training_start'],
                         training_end=m['training_end'], seconds=time.monotonic()-began))
    return pd.concat(rows), fits


def paired(reference, candidate, draws=2000):
    """Paired seven-day moving blocks; BA recomputed from confusion counts."""
    if not reference.index.equals(candidate.index):
        raise ValueError('Paired predictions must have identical timestamps')
    y = reference.actual.to_numpy().reshape(-1, 48)
    def counts(frame):
        p = frame.prediction.to_numpy().reshape(-1, 48)
        return np.stack([((y == k) & (p == j)).sum(axis=1)
                         for k,j in [(0,0),(0,1),(1,0),(1,1)]], axis=1)
    a,b = counts(reference),counts(candidate)
    def score(c):
        den = np.array([c[0]+c[1], c[2]+c[3]])
        return 100*np.mean(np.array([c[0],c[3]])[den>0]/den[den>0])
    rng = np.random.default_rng(SEED)
    delta=[]
    n=len(y); block=min(7,n)
    for _ in range(draws):
        starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        ix=(starts[:,None]+np.arange(block)).ravel()[:n]
        delta.append(score(b[ix].sum(axis=0))-score(a[ix].sum(axis=0)))
    return dict(delta_BA_pp=score(b.sum(axis=0))-score(a.sum(axis=0)),
                low=float(np.percentile(delta,2.5)),high=float(np.percentile(delta,97.5)))


def reports(name, p):
    days=market_days(p.index)
    groups=[('development',days.year<=2022),('confirmation',days.year>=2023)]
    groups += [(str(v),days.year==v) for v in sorted(set(days.year))]
    groups += [(str(v),days.to_period('M')==v) for v in days.to_period('M').unique()]
    return [dict(candidate=name,period=label,rows=int(mask.sum()),
                 **metrics(p.loc[mask,'actual'],p.loc[mask,'prediction']))
            for label,mask in groups if mask.any()]


