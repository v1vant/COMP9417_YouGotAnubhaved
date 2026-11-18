"""Round-2 screens (planning probes 1-6), reproduced for provenance.

Single-cutoff screens fit on market days before 2024-01-01 and score 2024-2025
(P0's own 2024 model covers the same rows). Only 2024-2025 was used to choose
round-2 candidates; 2015-2022 stays unseen by these screens. Probe 1 and the
contextual stacker use saved out-of-time predictions only.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ROOT, DATA, market_days, balanced_accuracy as ba, oracle
import live_horizon as lh

R = ROOT/'results'
BASE = dict(objective='binary', n_estimators=600, learning_rate=.04, num_leaves=31, min_child_samples=160,
            colsample_bytree=.85, subsample=.85, subsample_freq=1, reg_lambda=8., max_bin=127, n_jobs=8,
            verbosity=-1, deterministic=True, force_col_wise=True, random_state=9417, class_weight='balanced')
WIDE = dict(num_leaves=63, learning_rate=.02, n_estimators=1200, colsample_bytree=.5, min_child_samples=300)


def probe_cutoff():
    from research import paired
    rows = []
    def at(p, t):
        q = p.copy(); q['prediction'] = (q.probability > t).astype(int); return q
    for name in ['V0', 'V0b', 'V0c', 'bag3', 'V1_hl2', 'V2_hl4']:
        p = lh.load_predictions(R/'price_v3_live_horizon', name, 'live')
        days = market_days(p.index)
        d = paired(at(p, .57), at(p, .50))
        rows.append(dict(setting='live_horizon', variant=name, BA_050=ba(p.actual, p.probability > .5),
                         BA_057=ba(p.actual, p.probability > .57), delta=d['delta_BA_pp'], low=d['low'], high=d['high'],
                         sub_2023_24=ba(p.actual[days < '2025-01-01'], p.probability[days < '2025-01-01'] > .5)
                         - ba(p.actual[days < '2025-01-01'], p.probability[days < '2025-01-01'] > .57),
                         sub_2025_26=ba(p.actual[days >= '2025-01-01'], p.probability[days >= '2025-01-01'] > .5)
                         - ba(p.actual[days >= '2025-01-01'], p.probability[days >= '2025-01-01'] > .57)))
    p = lh.load_predictions(R/'price_v3_biennial', 'V0', 'biennial')
    years = market_days(p.index).year
    for label, mask in [('development_2015_2022', years <= 2022), ('confirmation_2023_2024', (years >= 2023) & (years <= 2024)),
                        ('reused_2025_2026', years >= 2025)] + [(str(y), years == y) for y in sorted(set(years))]:
        q = p.loc[mask]
        d = paired(at(q, .57), at(q, .50))
        rows.append(dict(setting='biennial_P0', variant=label, BA_050=ba(q.actual, q.probability > .5),
                         BA_057=ba(q.actual, q.probability > .57), delta=d['delta_BA_pp'], low=d['low'], high=d['high']))
    return pd.DataFrame(rows)


def probe_context_stacker(x, cols, idx):
    from lightgbm import LGBMClassifier
    p = lh.load_predictions(R/'price_v3_biennial', 'V0', 'biennial')
    pos = idx.get_indexer(p.index)
    want = ['slot', 'weekday', 'month', 'price_margin_1', 'price_margin_7', 'nsw_price_previous_std',
            'price_threshold', 'nsw_price_weekly_margin']
    raw = pd.DataFrame(np.asarray(x[pos][:, [cols.index(c) for c in want]]), index=p.index, columns=want)
    prob = np.clip(p.probability.to_numpy(), 1e-4, 1 - 1e-4)
    L = np.log(prob / (1 - prob)).reshape(-1, 48)
    f = {'l': L.ravel()}
    for k in (1, 2, 4):
        f[f'l_m{k}'] = L[:, np.maximum(np.arange(48) - k, 0)].ravel()
        f[f'l_p{k}'] = L[:, np.minimum(np.arange(48) + k, 47)].ravel()
    f['day_mean'], f['day_std'] = np.repeat(L.mean(1), 48), np.repeat(L.std(1), 48)
    for q in (10, 50, 90):
        f[f'day_q{q}'] = np.repeat(np.percentile(L, q, axis=1), 48)
    f['k05'] = np.repeat((L > 0).sum(1), 48)
    f['rank'] = (np.argsort(np.argsort(-L, axis=1), axis=1) / 47).ravel()
    F = pd.concat([pd.DataFrame(f, index=p.index), raw], axis=1)
    y, years = p.actual.to_numpy(), market_days(p.index).year
    rows = []
    for test in range(2019, 2027):
        tr, te = (years >= 2015) & (years < test), years == test
        m = LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=.03, min_child_samples=400,
                           colsample_bytree=.8, subsample=.8, subsample_freq=1, reg_lambda=10,
                           class_weight='balanced', n_jobs=8, verbosity=-1, random_state=1).fit(F[tr], y[tr])
        rows.append(dict(year=test, P0_050=ba(y[te], p.probability[te] > .5),
                         stacker_050=ba(y[te], m.predict_proba(F[te])[:, 1] > .5)))
    return pd.DataFrame(rows).assign(delta=lambda d: d.stacker_050 - d.P0_050)


def single_cutoff_screens(x, cols, idx, y):
    from lightgbm import LGBMClassifier, LGBMRegressor
    from model import load_data, feature_frame
    from recipe import weather_shape_names, abs_margin
    frame = load_data(DATA)
    shape = feature_frame(frame, weather_shape=True).loc[idx, weather_shape_names()].to_numpy(np.float32)
    days = market_days(frame.index)
    lookup = pd.Series(days, index=frame.index)
    more = {}
    for col in ('mel_temp', 'mel_solar', 'bne_temp', 'bne_solar'):   # weather shape for the other cities
        v = frame[col]
        daily = v.groupby(days).agg(['mean', 'min', 'max'])
        for off in (2, 4):
            b = v.groupby(days).shift(off).groupby(days).bfill()
            a = v.groupby(days).shift(-off).groupby(days).ffill()
            r = v - b
            more.update({f'{col}_before_{off}': b, f'{col}_after_{off}': a, f'{col}_ramp_{off}': r,
                         f'{col}_ramp_{off}_change_1d': r - r.shift(48)})
        for st in ('mean', 'min', 'max'):
            more[f'{col}_daily_{st}_change_1d'] = lookup.map(daily[st] - daily[st].shift(1))
        peak = v.groupby(days).agg(lambda g: int(np.argmax(g.to_numpy())))
        more[f'{col}_peak_slot'] = lookup.map(peak)
        more[f'{col}_peak_slot_change_1d'] = lookup.map(peak - peak.shift(1))
    cities = pd.DataFrame(more, index=frame.index).loc[idx].to_numpy(np.float32)
    price = frame.nsw_price
    means = price.groupby(days).agg(lambda s: s.mean())
    tau = lookup.map(means.shift(1)).loc[idx].to_numpy()
    uniq = market_days(idx).unique()
    def profile(col, lag):
        m = pd.DataFrame(frame[col].to_numpy().reshape(-1, 48), index=days.unique()).shift(lag).reindex(uniq).to_numpy()
        return np.repeat(m, 48, axis=0) - tau[:, None]
    regional = np.hstack([profile('vic_price', 1), profile('qld_price', 1), profile('nsw_price', 2)]).astype(np.float32)
    trend = ((market_days(idx) - pd.Timestamp('2010-01-01')).days.to_numpy() / 365.25).astype(np.float32)[:, None]
    margin = abs_margin(frame).loc[idx].to_numpy()
    d = market_days(idx)
    tr, te = d < pd.Timestamp('2024-01-01'), (d >= pd.Timestamp('2024-01-01')) & (d < pd.Timestamp('2026-01-01'))
    X = np.asarray(x)
    mono = [1 if (c.startswith('price_margin_') or c == 'nsw_price_weekly_margin'
                  or (c.startswith('nsw_price_') and '_before_margin_' in c)) else 0 for c in cols]
    def near(scale):
        w = 1 - np.exp(-margin[tr] / scale)
        t = y[tr]
        total = w.sum()
        for k in (0, 1):
            w[t == k] *= total / (2 * w[t == k].sum())
        return w / w.mean()
    screens = [
        ('V0_refit', None, {}, None), ('+trend', trend, {}, None), ('+weather_shape', shape, {}, None),
        ('+vic_qld_nsw2_profiles', regional, {}, None), ('colsample0.4', None, dict(colsample_bytree=.4), None),
        ('wide_slow', None, WIDE, None), ('V0_seed9418', None, dict(random_state=9418), None),
        ('+weather_shape_seed9418', shape, dict(random_state=9418), None),
        ('+weather_shape+3city_shape', np.hstack([shape, cities]), {}, None),
        ('+weather_shape+wide_slow', shape, WIDE, None),
        ('monotone_margins', None, dict(monotone_constraints=mono, monotone_constraints_method='advanced'), None),
        ('downweight_near_0.10', None, dict(class_weight=None), .10),
        ('downweight_near_0.25', None, dict(class_weight=None), .25),
    ]
    a, years = y[te], d[te].year
    rows = []
    for name, extra, over, scale in screens:
        began = time.monotonic()
        xtr = X[tr] if extra is None else np.hstack([X[tr], extra[tr]])
        xte = X[te] if extra is None else np.hstack([X[te], extra[te]])
        model = LGBMClassifier(**{**BASE, **over}).fit(xtr, y[tr], sample_weight=None if scale is None else near(scale))
        p = model.predict_proba(xte)[:, 1]
        for label, m in [('2024', years == 2024), ('2025', years == 2025), ('2024-2025', np.ones(len(a), bool))]:
            rows.append(dict(screen=name, period=label, features=xtr.shape[1], AUC=100*roc_auc_score(a[m], p[m]),
                             BA_050=ba(a[m], p[m] > .5), oracle_BA=oracle(a[m], p[m])[0],
                             seconds=time.monotonic() - began))
        print(rows[-1], flush=True)
    # Probe 3: asinh margin regression (Huber), ranking only.
    from scipy.stats import rankdata
    target = np.arcsinh(np.sign(np.where(y == 1, 1, -1)) * margin)   # sign(P - tau) * |z|
    reg = LGBMRegressor(objective='huber', alpha=1.0, **{k: v for k, v in BASE.items()
                                                         if k not in ('objective', 'class_weight')})
    z = reg.fit(X[tr], target[tr]).predict(X[te])
    p0 = pd.read_parquet(R/'price_v3_biennial/fits/V0_2024-01-01.parquet').probability.to_numpy()
    def rank(v):
        return rankdata(v) / len(v)
    for label, m in [('2024', years == 2024), ('2025', years == 2025), ('2024-2025', np.ones(len(a), bool))]:
        rows.append(dict(screen='margin_regressor', period=label, AUC=100*roc_auc_score(a[m], z[m]),
                         oracle_BA=oracle(a[m], rank(z[m]))[0], BA_050=ba(a[m], z[m] > 0)))
        rows.append(dict(screen='rank_blend_P0_regressor_0.5', period=label,
                         oracle_BA=oracle(a[m], .5*rank(p0[m]) + .5*rank(z[m]))[0]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=R/'price_v3_screen')
    ap.add_argument('--cache', type=Path, required=True, help='live_horizon.py feature cache')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.load(args.cache/'x.npy', mmap_mode='r')
    cols = json.loads((args.cache/'columns.json').read_text())
    idx = pd.DatetimeIndex(np.load(args.cache/'index.npy'))
    y = np.load(args.cache/'price.npy').astype(int)
    probe_cutoff().to_csv(args.output/'probe1_cutoff.csv', index=False)
    probe_context_stacker(x, cols, idx).to_csv(args.output/'probe2_context_stacker.csv', index=False)
    single_cutoff_screens(x, cols, idx, y).to_csv(args.output/'probes3_6_single_cutoff.csv', index=False)
    print('Saved screens to', args.output)


if __name__ == '__main__':
    main()
