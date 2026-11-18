"""Round-2 selection: apply the rule frozen in the plan BEFORE the C1-C3 runs.

Eligible (each vs C0_bag3, all bags of seeds 9417/9418/9419, cutoff 0.50):
  live horizon: pooled BA delta > 0, >= 10/18 windows improved, AUC delta > 0 in both sub-periods;
  biennial development 2015-2022 (unseen by the screens): BA delta >= 0 and AUC delta > 0.
Choose the eligible candidate with the highest pooled live-horizon BA, else C0.
Cutoff stays 0.50 unless 0.57 is better for the chosen candidate in BOTH live sub-periods.
"""
import argparse
import json
from pathlib import Path
import time

import pandas as pd

from common import ROOT
from live_horizon import ROUND2, SEEDS, SUBPERIODS, VARIANTS

R = ROOT/'results'


def load(folder, cutoff):
    path = folder/f'report_cutoff_{cutoff:.2f}'
    return pd.read_csv(path/'metrics.csv'), pd.read_csv(path/'paired_deltas.csv')


def value(metrics, variant, period, column):
    row = metrics[(metrics.variant == variant) & (metrics.period == period)]
    if len(row) != 1:
        raise KeyError(f'{variant}/{period} missing')
    return float(row[column].iloc[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--live', type=Path, default=R/'price_v3_live_horizon')
    ap.add_argument('--biennial', type=Path, default=R/'price_v3_biennial')
    ap.add_argument('--output', type=Path, default=R/'price_v3_final'/'frozen_selection_round2.json')
    args = ap.parse_args()
    live, live_deltas = load(args.live, .50)
    bien, bien_deltas = load(args.biennial, .50)
    windows = sorted(p for p in live.period.unique() if p.startswith('window_'))
    if len(windows) != 18:
        raise ValueError('Expected 18 live-horizon windows')
    control = 'C0_bag3'
    rows = []
    for family in ROUND2:
        name = f'{family}_bag3'
        if name not in set(live.variant) or name not in set(bien.variant):
            rows.append(dict(candidate=name, evaluated=False))
            continue
        d_live = value(live, name, 'pooled', 'BA') - value(live, control, 'pooled', 'BA')
        improved = sum(value(live, name, w, 'BA') > value(live, control, w, 'BA') for w in windows)
        d_auc = [value(live, name, l, 'AUC') - value(live, control, l, 'AUC') for l, _, _ in SUBPERIODS]
        d_sub = [value(live, name, l, 'BA') - value(live, control, l, 'BA') for l, _, _ in SUBPERIODS]
        dev = 'development_2015_2022'
        d_dev = value(bien, name, dev, 'BA') - value(bien, control, dev, 'BA')
        d_dev_auc = value(bien, name, dev, 'AUC') - value(bien, control, dev, 'AUC')
        ci = live_deltas[(live_deltas.variant == name) & (live_deltas.period == 'pooled')].iloc[0]
        dev_ci = bien_deltas[(bien_deltas.variant == name) & (bien_deltas.period == dev)].iloc[0]
        eligible = d_live > 0 and improved >= 10 and min(d_auc) > 0 and d_dev >= 0 and d_dev_auc > 0
        rows.append(dict(candidate=name, evaluated=True, live_pooled_BA=value(live, name, 'pooled', 'BA'),
                         live_delta_pp=d_live, live_ci=[float(ci.low), float(ci.high)], windows_improved=improved,
                         live_subperiod_BA_delta_pp=d_sub, live_subperiod_AUC_delta_pp=d_auc,
                         development_delta_pp=d_dev, development_ci=[float(dev_ci.low), float(dev_ci.high)],
                         development_AUC_delta_pp=d_dev_auc,
                         confirmation_delta_pp=value(bien, name, 'confirmation_2023_2024', 'BA')
                         - value(bien, control, 'confirmation_2023_2024', 'BA'),
                         reused_2025_2026_delta_pp=value(bien, name, 'reused_2025_2026', 'BA')
                         - value(bien, control, 'reused_2025_2026', 'BA'),
                         eligible=bool(eligible)))
    eligible = [r for r in rows if r.get('eligible')]
    chosen = max(eligible, key=lambda r: r['live_pooled_BA'])['candidate'] if eligible else control
    live57, _ = load(args.live, .57)
    better57 = [value(live57, chosen, l, 'BA') > value(live, chosen, l, 'BA') for l, _, _ in SUBPERIODS]
    cutoff = .57 if all(better57) else .50
    family = chosen.split('_')[0]
    config = ({} if family == 'C0' else
              {k: v for k, v in VARIANTS[f'{family}_s{SEEDS[0]}'].items() if k not in ('recipe', 'threshold', 'seed')})
    decision = dict(
        round=2, frozen=time.strftime('%Y-%m-%d %H:%M:%S'), chosen=chosen, cutoff=cutoff,
        cutoff_rule='0.50 unless 0.57 better in both live sub-periods', cutoff_057_better_by_subperiod=better57,
        control=dict(name=control, live_pooled_BA=value(live, control, 'pooled', 'BA'),
                     biennial_development_BA=value(bien, control, 'development_2015_2022', 'BA')),
        candidates=rows,
        package_candidate=dict(name=f'{chosen}_cutoff{cutoff:.2f}', threshold=cutoff, half_life_years=None,
                               seeds=[9417, 9418, 9419, 9420, 9421], **config,
                               source='code/price_v3/choose_candidate.py (round 2)'),
        note=('Bag size 5 if the measured rebuild is <= 75 min on 4 cores, else seeds[:3]. '
              '2023-2026 is a reused period; 2015-2022 was unseen by the round-2 screens; the live week is the only '
              'unseen test. Demand recipe unchanged.'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, default=float))
    print(json.dumps(decision, indent=2, default=float))


if __name__ == '__main__':
    main()
