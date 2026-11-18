"""WP1: live-horizon Stage-3 final-recipe validation.

Each cutoff is an INDEPENDENT check of a training recipe at the live-week
staleness (model trained through a month end, forecasting the next 1-2 months),
not a deployed adaptation sequence. Nov 2023-Oct 2026 has been inspected before;
it is a reused historical period, and the live week is the only unseen test.
Variants are pre-registered in frozen_variants.json before any fit is scored.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import tempfile
from pathlib import Path
import time

import numpy as np
import pandas as pd

from common import ROOT, DATA, THRESHOLD, market_days, balanced_accuracy, oracle, harness_targets, weekly_gain

BASE = dict(family='boosted', leaves=31, trees=600, learning_rate=.04, threshold=THRESHOLD,
            price_feature_set='base', compact=False)
VARIANTS = {
    'V0': dict(BASE, seed=9417),
    'V0b': dict(BASE, seed=9418),
    'V0c': dict(BASE, seed=9419),
    'V1_hl2': dict(BASE, seed=9417, half_life_years=2.),
    'V2_hl4': dict(BASE, seed=9417, half_life_years=4.),
    # WP2 stacked arm and its matched control (level OOF starts in 2011).
    'L0_ctrl2011': dict(BASE, seed=9417, training_start='2011-01-01'),
    'L1_level': dict(BASE, seed=9417, training_start='2011-01-01', extra='LEVEL'),
}
MATCHED = {'L1_level': 'L0_ctrl2011'}
BAGS = {'bag3': ['V0', 'V0b', 'V0c']}
# Round 2 (pre-registered): shared v3 recipe; seed-matched against V0/V0b/V0c, bags against C0_bag3.
SEEDS = (9417, 9418, 9419)
WIDE = dict(num_leaves=63, learning_rate=.02, n_estimators=1200, colsample_bytree=.5, min_child_samples=300)
ROUND2 = {
    'C1': dict(recipe='v3', threshold=THRESHOLD, weather_shape=True, params={}),
    'C2': dict(recipe='v3', threshold=THRESHOLD, weather_shape=True, params=WIDE),
    'C3': dict(recipe='v3', threshold=THRESHOLD, weather_shape=True, params=WIDE, near_threshold_scale=.10),
}
BAGS['C0_bag3'] = ['V0', 'V0b', 'V0c']
for _family, _config in ROUND2.items():
    for _seed, _control in zip(SEEDS, ['V0', 'V0b', 'V0c']):
        VARIANTS[f'{_family}_s{_seed}'] = dict(_config, seed=_seed)
        MATCHED[f'{_family}_s{_seed}'] = _control
    BAGS[f'{_family}_bag3'] = [f'{_family}_s{_seed}' for _seed in SEEDS]
    MATCHED[f'{_family}_bag3'] = 'C0_bag3'
CUTOFFS = list(pd.date_range('2023-11-01', '2026-09-01', freq='2MS'))
SUBPERIODS = [('2023-11_2024-12', '2023-11-01', '2025-01-01'), ('2025-01_2026-10', '2025-01-01', '2026-11-01')]
CONTROL = 'V0'
# Budget-compliant schedule used by P0: initial 2015 fit + replacements 2016..2024 and 2026.
BIENNIAL_STARTS = pd.to_datetime(['2015-01-01', '2016-01-01', '2018-01-01', '2020-01-01', '2022-01-01',
                                  '2024-01-01', '2026-01-01', '2026-11-01'])
BIENNIAL_PERIODS = [('development_2015_2022', '2015-01-01', '2023-01-01'),
                    ('confirmation_2023_2024', '2023-01-01', '2025-01-01'),
                    ('reused_2025_2026', '2025-01-01', '2026-11-01')]


def window(cutoff):
    return cutoff, cutoff + pd.DateOffset(months=2)


def schedule(kind):
    if kind == 'live':
        return [window(c) for c in CUTOFFS]
    if kind == 'biennial':
        return list(zip(BIENNIAL_STARTS[:-1], BIENNIAL_STARTS[1:]))
    raise ValueError(kind)


# ---------------------------------------------------------------- preparation
def prepare_v3(cache):
    """Weather-shape columns and |z| margins aligned to the cached rows (disposable)."""
    if (cache/'ws.npy').exists() and (cache/'margin.npy').exists():
        return
    from model import load_data, feature_frame
    from recipe import weather_shape_names, abs_margin
    frame = load_data(DATA)
    index = pd.DatetimeIndex(np.load(cache/'index.npy'))
    shape = feature_frame(frame, weather_shape=True).loc[index, weather_shape_names()].to_numpy(dtype=np.float32)
    margin = abs_margin(frame).loc[index].to_numpy(dtype=np.float64)
    if not (np.isfinite(shape).all() and np.isfinite(margin).all()):
        raise ValueError('Nonfinite weather-shape or margin cache values')
    np.save(cache/'ws.npy', shape)
    np.save(cache/'margin.npy', margin)


def prepare(cache, output):
    """Build the 571-feature P0 table once; workers memory-map it (disposable cache)."""
    cache.mkdir(parents=True, exist_ok=True)
    if (cache/'done.json').exists() and (output/'targets.parquet').exists():
        prepare_v3(cache)
        return
    from model import load_data
    from pipeline import table
    frame = load_data(DATA)
    if market_days(frame.index).max() != pd.Timestamp('2026-10-31'):
        raise ValueError('Expected released data through market day 2026-10-31')
    x, y = table(frame)
    if x.shape[1] != 571:
        raise ValueError(f'Expected the 571-feature P0 schema, got {x.shape[1]}')
    targets = harness_targets(frame).loc[x.index]
    if not (targets.actual == y.price).all():
        raise ValueError('Training labels differ from harness targets')
    np.save(cache/'x.npy', np.ascontiguousarray(x.to_numpy(dtype=np.float32)))
    np.save(cache/'index.npy', x.index.to_numpy())  # keeps the datetime64 unit
    np.save(cache/'price.npy', y.price.to_numpy(dtype=np.int8))
    targets[['actual', 'yesterday', 'last_week']].to_parquet(output/'targets.parquet')
    (cache/'columns.json').write_text(json.dumps(list(x.columns)))
    (cache/'done.json').write_text(json.dumps(dict(rows=len(x), columns=x.shape[1],
                                                   start=str(x.index[0]), end=str(x.index[-1]))))
    prepare_v3(cache)


_STATE = {}


def _init(cache):
    from threadpoolctl import threadpool_limits
    _STATE['limits'] = threadpool_limits(limits=2)
    _STATE['x'] = np.load(cache/'x.npy', mmap_mode='r')
    _STATE['index'] = pd.DatetimeIndex(np.load(cache/'index.npy'))
    _STATE['price'] = np.load(cache/'price.npy')
    _STATE['columns'] = json.loads((cache/'columns.json').read_text())
    if (cache/'ws.npy').exists():
        from recipe import weather_shape_names
        _STATE['ws'] = np.load(cache/'ws.npy', mmap_mode='r')
        _STATE['ws_columns'] = weather_shape_names()
        _STATE['margin'] = np.load(cache/'margin.npy', mmap_mode='r')


def _frame(mask, extra, weather_shape=False):
    x = pd.DataFrame(np.asarray(_STATE['x'][mask]), index=_STATE['index'][mask],
                     columns=_STATE['columns'], copy=False)
    if weather_shape:
        x = pd.concat([x, pd.DataFrame(np.asarray(_STATE['ws'][mask]), index=x.index,
                                       columns=_STATE['ws_columns'])], axis=1)
    if extra is not None:
        x = pd.concat([x, extra.loc[x.index]], axis=1)
    return x


def run_one(name, config, cutoff, destination, stop=None):
    from price_train import fit_model
    from model import predict_price_probability
    began = time.monotonic()
    config = dict(config)
    extra = pd.read_parquet(config.pop('extra')) if config.get('extra') else None
    config.pop('extra', None)
    start = pd.Timestamp(config.pop('training_start')) if config.get('training_start') else None
    days = market_days(_STATE['index'])
    train = days < cutoff
    if start is not None:
        train &= days >= start
    if extra is not None:
        train &= np.asarray(_STATE['index'].isin(extra.index))
    lo, hi = window(cutoff) if stop is None else (cutoff, stop)
    score = (days >= lo) & (days < hi)
    shape = bool(config.get('weather_shape'))
    x = _frame(train, extra, shape)
    if config.get('recipe') == 'v3':
        from recipe import fit_price
        margin = np.asarray(_STATE['margin'][train]) if config.get('near_threshold_scale') else None
        estimator = fit_price(x, _STATE['price'][train], config['seed'], config.get('params'),
                              config.get('half_life_years'), config.get('near_threshold_scale'), margin)
        bundle = dict(training_start=str(market_days(x.index).min()), training_end=str(market_days(x.index).max()))
    else:
        y = pd.DataFrame({'price': _STATE['price'][train]}, index=x.index)
        bundle = fit_model('price', config, x, y)
        del y
    if pd.Timestamp(bundle['training_end']) >= cutoff:
        raise RuntimeError('Training data reached the cutoff')
    del x
    xs = _frame(score, extra, shape)
    probability = (estimator.predict_proba(xs)[:, 1] if config.get('recipe') == 'v3'
                   else predict_price_probability(bundle, xs))
    out = pd.DataFrame(dict(actual=_STATE['price'][score].astype(int), probability=probability,
                            prediction=(probability > config['threshold']).astype(int),
                            cutoff=cutoff), index=xs.index)
    if len(out) % 48 or market_days(out.index).min() != lo:
        raise RuntimeError('Incomplete scoring window')
    out.to_parquet(destination)
    return dict(variant=name, cutoff=str(cutoff.date()), training_start=bundle['training_start'],
                training_end=bundle['training_end'], train_rows=int(train.sum()), score_rows=len(out),
                seconds=time.monotonic() - began)


# ---------------------------------------------------------------- registration
def resolve(config, cutoff, kind, level_dir):
    """The level arm uses OOF rows on the biennial schedule, per-cutoff extras on the live one."""
    if config.get('extra') != 'LEVEL':
        return config
    path = level_dir/'oof_rows.parquet' if kind == 'biennial' else level_dir/'harness_extra'/f'{cutoff.date()}.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Run level.py first: {path}')
    return dict(config, extra=str(path))


def register(output, names, kind):
    path = output/'frozen_variants.json'
    frozen = json.loads(path.read_text()) if path.exists() else {}
    protocol = dict(schedule=kind, periods=[[str(a.date()), str(b.date())] for a, b in schedule(kind)],
                    threshold=THRESHOLD, control=CONTROL, matched_controls=MATCHED, bags=BAGS,
                    interpretation=('Stage-3 final-recipe checks on a reused historical period' if kind == 'live'
                                    else 'Budget-compliant biennial simulation (six replacements)'))
    if '_protocol' in frozen and frozen['_protocol'].get('schedule', 'live') != kind:
        raise ValueError('Output directory belongs to a different schedule')
    for name in names:
        if name in frozen:
            if frozen[name]['config'] != VARIANTS[name]:
                raise ValueError(f'{name} changed after registration; use a new variant name')
        else:
            frozen[name] = dict(config=VARIANTS[name], registered=time.strftime('%Y-%m-%d %H:%M:%S'))
    frozen['_protocol'] = protocol
    path.write_text(json.dumps(frozen, indent=2))


def run(output, cache, names, workers, kind='live', level_dir=None):
    prepare(cache, output)
    register(output, names, kind)
    (output/'fits').mkdir(exist_ok=True)
    tasks = [(n, a, b) for n in names for a, b in schedule(kind)
             if not (output/'fits'/f'{n}_{a.date()}.parquet').exists()]
    status = dict(state='running', schedule=kind, pending=len(tasks), done=0,
                  started=time.strftime('%Y-%m-%d %H:%M:%S'))
    def save():
        (output/'status.json').write_text(json.dumps(status, indent=2))
    save()
    log_path = output/'fits.csv'
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init, initargs=(cache,)) as pool:
        futures = {pool.submit(run_one, n, resolve(VARIANTS[n], a, kind, level_dir), a,
                               output/'fits'/f'{n}_{a.date()}.parquet', b): n for n, a, b in tasks}
        for future in as_completed(futures):
            record = future.result()
            pd.DataFrame([record]).to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
            status['done'] += 1
            status['last'] = record
            save()
            print(record, flush=True)
    status['state'] = 'fits_completed'
    save()


# ---------------------------------------------------------------- reporting
def load_predictions(output, name, kind='live', cutoff=THRESHOLD):
    """Saved probabilities, re-thresholded at ``cutoff`` (bags average member probabilities first)."""
    if name in BAGS:
        parts = [load_predictions(output, n, kind, cutoff) for n in BAGS[name]]
        p = parts[0].copy()
        p['probability'] = np.mean([q.probability.to_numpy() for q in parts], axis=0)
        p['prediction'] = (p.probability > cutoff).astype(int)
        return p
    files = [output/'fits'/f'{name}_{a.date()}.parquet' for a, _ in schedule(kind)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f'{name}: missing {missing}')
    p = pd.concat([pd.read_parquet(f) for f in files])
    if not p.index.is_unique or not p.index.is_monotonic_increasing:
        raise ValueError('Windows must be contiguous and non-overlapping')
    p['prediction'] = (p.probability > cutoff).astype(int)
    return p


def describe(p, mask=None, with_oracle=True):
    if mask is not None:
        p = p.loc[mask]
    from sklearn.metrics import roc_auc_score
    a, pr, pred = p.actual.to_numpy(), p.probability.to_numpy(), p.prediction.to_numpy()
    best, cut = oracle(a, pr) if with_oracle else (np.nan, np.nan)
    return dict(rows=len(p), BA=balanced_accuracy(a, pred), accuracy=100*np.mean(a == pred),
                recall_0=100*np.mean(pred[a == 0] == 0), recall_1=100*np.mean(pred[a == 1] == 1),
                prevalence_1=100*a.mean(), predicted_1=100*pred.mean(),
                AUC=100*roc_auc_score(a, pr) if len(set(a)) > 1 else np.nan,
                oracle_BA_diagnostic=best, oracle_cutoff_diagnostic=cut)


def control_for(name):
    return MATCHED.get(name, CONTROL)


def report(output, names, kind='live', cutoff=THRESHOLD):
    from research import paired
    targets = pd.read_parquet(output/'targets.parquet')
    folder = output/f'report_cutoff_{cutoff:.2f}'
    folder.mkdir(exist_ok=True)
    predictions = {n: load_predictions(output, n, kind, cutoff) for n in names}
    named = SUBPERIODS if kind == 'live' else BIENNIAL_PERIODS
    rows, deltas, weekly = [], [], []
    for name, p in predictions.items():
        reference = predictions.get(control_for(name), p)   # report-only subsets may omit the control
        if not p.index.equals(reference.index) or not p.actual.equals(reference.actual):
            raise ValueError(f'{name}: predictions are not aligned with its control')
        days = market_days(p.index)
        periods = [('pooled', np.ones(len(p), bool))]
        periods += [(label, (days >= pd.Timestamp(a)) & (days < pd.Timestamp(b))) for label, a, b in named]
        periods += [(f'window_{a.date()}', (days >= a) & (days < b)) for a, b in schedule(kind)]
        periods += [(str(y), days.year == y) for y in sorted(set(days.year))]
        periods += [(str(m), days.to_period('M') == m) for m in days.to_period('M').unique()]
        for label, mask in periods:
            detailed = label == 'pooled' or label.isdigit() or label in {l for l, _, _ in named}   # oracle: aggregates only
            rows.append(dict(variant=name, period=label, cutoff=cutoff, **describe(p, mask, detailed)))
        if name != control_for(name) and control_for(name) in predictions:
            for label, mask in [('pooled', None)] + [(l, (days >= pd.Timestamp(a)) & (days < pd.Timestamp(b)))
                                                     for l, a, b in named]:
                c, q = (reference, p) if mask is None else (reference.loc[mask], p.loc[mask])
                deltas.append(dict(variant=name, control=control_for(name), period=label, **paired(c, q)))
        g = weekly_gain(p.actual, p.prediction, targets.yesterday.loc[p.index])
        weekly.append(g.assign(variant=name))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(folder/'metrics.csv', index=False)
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(folder/'paired_deltas.csv', index=False)
    weekly = pd.concat(weekly)
    weekly.to_csv(folder/'weekly_gain.csv', index=False)
    gate = []
    for name in names:
        if name == control_for(name) or control_for(name) not in predictions:
            continue
        base = metrics[metrics.variant == control_for(name)].set_index('period')
        m = metrics[metrics.variant == name].set_index('period')
        d = deltas[deltas.variant == name].set_index('period')
        if kind == 'live':
            windows = [f'window_{a.date()}' for a, _ in schedule(kind)]
            positive = int((m.loc[windows, 'BA'] > base.loc[windows, 'BA']).sum())
            sub = [d.loc[l, 'delta_BA_pp'] for l, _, _ in SUBPERIODS]
            accuracy = m.loc['pooled', 'accuracy'] - base.loc['pooled', 'accuracy']
            gate.append(dict(variant=name, control=control_for(name), delta_BA_pp=d.loc['pooled', 'delta_BA_pp'],
                             low=d.loc['pooled', 'low'], high=d.loc['pooled', 'high'],
                             subperiod_deltas=';'.join(f'{v:.3f}' for v in sub), positive_windows=positive,
                             accuracy_delta_pp=accuracy,
                             passes_live_horizon_gate=bool(d.loc['pooled', 'delta_BA_pp'] >= .3 and d.loc['pooled', 'low'] > 0
                                                           and min(sub) >= 0 and positive >= 11 and accuracy >= -.25)))
        else:
            dev = 'development_2015_2022'
            years = [str(y) for y in range(2015, 2023)]
            positive = int((m.loc[years, 'BA'] > base.loc[years, 'BA']).sum())
            accuracy = m.loc[dev, 'accuracy'] - base.loc[dev, 'accuracy']
            gate.append(dict(variant=name, control=control_for(name), development_delta_pp=d.loc[dev, 'delta_BA_pp'],
                             low=d.loc[dev, 'low'], high=d.loc[dev, 'high'], positive_development_years=positive,
                             development_accuracy_delta_pp=accuracy,
                             confirmation_delta_pp=d.loc['confirmation_2023_2024', 'delta_BA_pp'],
                             reused_2025_2026_delta_pp=d.loc['reused_2025_2026', 'delta_BA_pp'],
                             passes_development_gate=bool(d.loc[dev, 'delta_BA_pp'] >= .25 and d.loc[dev, 'low'] > 0
                                                          and positive >= 6 and accuracy >= -.25
                                                          and d.loc['confirmation_2023_2024', 'delta_BA_pp'] > 0),
                             no_development_harm=bool(d.loc[dev, 'delta_BA_pp'] >= -.1)))
    gate = pd.DataFrame(gate)
    gate.to_csv(folder/'gate.csv', index=False)
    summary = metrics[metrics.period.isin(['pooled'] + [l for l, _, _ in named])]
    weekly_summary = weekly.groupby('variant').G.agg(weeks='size', median='median', p10=lambda s: s.quantile(.1),
                                                     p90=lambda s: s.quantile(.9),
                                                     share_ge_8=lambda s: 100*(s >= 8).mean())
    weekly_summary.to_csv(folder/'weekly_gain_summary.csv')
    pd.set_option('display.width', 250)
    print(summary.round(3).to_string(index=False))
    print(gate.round(3).to_string(index=False))
    print(weekly_summary.round(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'results'/'price_v3_live_horizon')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--cache', type=Path, default=Path(tempfile.gettempdir())/'price_v3_cache',
                    help='Disposable memory-mapped feature cache (about 0.7 GB); kept out of results/')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--schedule', choices=['live', 'biennial'], default='live')
    ap.add_argument('--level', type=Path, default=ROOT/'results'/'price_v3_level')
    ap.add_argument('--cutoffs', type=float, nargs='+', default=[THRESHOLD],
                    help='Report every variant at these probability cutoffs (one folder each)')
    args = ap.parse_args()
    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        ap.error(f'Unknown variants {unknown}')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run(args.output, args.cache, args.variants, args.workers, args.schedule, args.level)
    names = list(args.variants)
    names += [b for b, parts in BAGS.items() if all(p in names for p in parts) and b not in names]
    for cutoff in args.cutoffs:
        report(args.output, names, args.schedule, cutoff)


if __name__ == '__main__':
    main()
