"""Evaluate ONE frozen candidate against P0 on January 2025–October 2026.

This is a reused historical check, not a new untouched holdout. No tuning or
candidate selection uses its scores. Does not change submission artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pipeline import table, calibration
from research import ROOT, fit, threshold, metrics, paired, load_data, market_days, predict_price_probability


def evaluate_policy(frame, policy, output):
    """At 2025 reuse the January-2024 model; replace it once in January 2026."""
    if policy.get('components'):
        raise ValueError('Use evaluate_ensemble for a mixture')
    c = policy['config']
    x, y = table(frame, compact=c.get('compact', False), context=c.get('price_feature_set') == 'both')
    if policy.get('stacked'):
        from stack import generate
        print('Generating chronological demand features (MLP logs in demand_oof/).', flush=True)
        extra = generate(frame, x, y, output / 'demand_oof')
        x = pd.concat([x.loc[extra.index], extra], axis=1)
        y = y.loc[x.index]
    elif policy.get('training_start'):
        keep = market_days(x.index) >= pd.Timestamp(policy['training_start'])
        x, y = x.loc[keep], y.loc[keep]
    days = market_days(x.index)
    forecasts, audits = [], []
    t = policy.get('_previous_threshold', c.get('threshold', .57))
    for effective, score_start, score_stop in [('2024-01-01', '2025-01-01', '2026-01-01'),
                                               ('2026-01-01', '2026-01-01', '2026-11-01')]:
        cutoff = pd.Timestamp(effective)
        if policy.get('refresh'):
            probs = calibration(c, x, y, cutoff)
            assert market_days(probs.index).max() < cutoff
            t = threshold(y.loc[probs.index, 'price'], probs.to_numpy(), t)
            pd.DataFrame(dict(probability=probs, actual=y.loc[probs.index, 'price'])).to_parquet(
                output / f'calibration_{cutoff.year}.parquet')
        print(f'Fitting price model through {(cutoff-pd.Timedelta(days=1)).date()}', flush=True)
        model = fit(c, x, y, cutoff)
        scoring = (days >= pd.Timestamp(score_start)) & (days < pd.Timestamp(score_stop))
        probabilities = predict_price_probability(model, x.loc[scoring])
        forecasts.append(pd.DataFrame(dict(actual=y.loc[scoring, 'price'], probability=probabilities,
                                           prediction=(probabilities > t).astype(int), threshold=t)))
        audits.append(dict(effective_day=effective, training_start=model['training_start'],
                           training_end=model['training_end'], score_start=score_start, score_stop_exclusive=score_stop,
                           threshold=t, replacement=effective == '2026-01-01'))
        pd.DataFrame(audits).to_csv(output / 'fits.csv', index=False)
    result = pd.concat(forecasts)
    if len(result) != 669 * 48 or not result.index.is_unique:
        raise ValueError('Expected all 669 complete holdout days')
    return result


def evaluate_ensemble(frame, policy, configurations, output):
    components = []
    for name in policy['components']:
        folder = output / name
        folder.mkdir()
        component_policy = dict(configurations[name], refresh=True)
        components.append(evaluate_policy(frame, component_policy, folder))
    a, b = components
    if not a.index.equals(b.index):
        raise ValueError('Ensemble prediction dates differ')
    w = policy['weight']
    result = a.copy()
    result['probability'] = w*a.probability + (1-w)*b.probability
    t = policy.get('_previous_threshold', .57)
    for year in (2024, 2026):
        ca = pd.read_parquet(output / policy['components'][0] / f'calibration_{year}.parquet')
        cb = pd.read_parquet(output / policy['components'][1] / f'calibration_{year}.parquet')
        if not ca.index.equals(cb.index):
            raise ValueError('Ensemble calibration dates differ')
        t = threshold(ca.actual, (w*ca.probability + (1-w)*cb.probability).to_numpy(), t)
        mask = market_days(result.index).year == (2025 if year == 2024 else 2026)
        result.loc[mask, 'threshold'] = t
    result['prediction'] = (result.probability > result.threshold).astype(int)
    return result


def period_rows(name, p):
    days = market_days(p.index)
    periods = [('holdout', np.ones(len(p), dtype=bool))]
    periods += [(str(year), days.year == year) for year in (2025, 2026)]
    periods += [(str(month), days.to_period('M') == month) for month in days.to_period('M').unique()]
    return [dict(candidate=name, period=label, rows=int(mask.sum()),
                 **metrics(p.loc[mask, 'actual'], p.loc[mask, 'prediction'])) for label, mask in periods]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--research', type=Path, default=ROOT/'results/price_v2')
    ap.add_argument('--candidate', default='stacked')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--data', type=Path, default=ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    args = ap.parse_args()
    configs = json.loads((args.research/'configurations.json').read_text())
    status = json.loads((args.research/'status.json').read_text())
    if status.get('quick') or status['state'] != 'completed':
        ap.error('Requires a completed, full research run')
    if args.candidate not in configs or args.candidate == 'P0':
        ap.error('Choose one saved challenger; P0 is always evaluated as the control')
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('Use a new empty output directory; existing results will not be overwritten')
    # Preserve the historically deployed cutoff when ties occur during refresh.
    for name, policy in configs.items():
        path = args.research/f'{name}_predictions.parquet'
        if path.exists():
            past = pd.read_parquet(path, columns=['threshold'])
            policy['_previous_threshold'] = float(past.threshold.iloc[-1])
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frozen = dict(candidate=args.candidate, policy=configs[args.candidate], control=configs['P0'],
                  interpretation='Previously inspected historical period; no candidate selection on these scores',
                  component_policies={n: configs[n] for n in configs[args.candidate].get('components', [])},
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*.py')})
    (args.output/'frozen_selection.json').write_text(json.dumps(frozen, indent=2))
    def progress(state):
        (args.output/'status.json').write_text(json.dumps(dict(state=state, candidate=args.candidate,
            seconds=time.monotonic()-started), indent=2))
    progress('loading_data')
    try:
        frame = load_data(args.data)
        if market_days(frame.index).max() != pd.Timestamp('2026-10-31'):
            raise ValueError('Expected released data through market day 2026-10-31')
        results, rows = {}, []
        for name in ['P0', args.candidate]:
            progress('running_'+name)
            folder = args.output/name
            folder.mkdir()
            policy = configs[name]
            p = (evaluate_ensemble(frame, policy, configs, folder) if policy.get('components') else
                 evaluate_policy(frame, policy, folder))
            if results and not p.actual.equals(results['P0'].actual):
                raise ValueError('Control and candidate labels must match exactly')
            results[name] = p
            p.to_parquet(args.output/f'{name}_predictions.parquet')
            rows.extend(period_rows(name, p))
            pd.DataFrame(rows).to_csv(args.output/'metrics.csv', index=False)
            print(name, metrics(p.actual, p.prediction), flush=True)
        comparison = paired(results['P0'], results[args.candidate])
        (args.output/'paired_comparison.json').write_text(json.dumps(comparison, indent=2))
        print('Paired balanced-accuracy change:', comparison, flush=True)
        progress('completed')
    except Exception as exc:
        progress('failed: '+str(exc))
        raise

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
