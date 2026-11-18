"""WP0: diagnostics from SAVED predictions only (no fitting).

Separates ranking quality from cutoff choice, decomposes P0 errors into daily
count (level) and within-day shape, profiles regime-specific errors, and gives
the weekly gain over the yesterday baseline that drives the live-week mark.
2025-2026 is a reused, previously inspected period; nothing here selects a model.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ROOT, DATA, THRESHOLD, market_days, balanced_accuracy, oracle, harness_targets, weekly_gain

R = ROOT / 'results'


def joined(*paths, rename=None):
    parts = [pd.read_parquet(p) for p in paths]
    frame = pd.concat(parts)
    frame = frame[~frame.index.duplicated(keep='first')].sort_index()
    return frame.rename(columns=rename or {})


def candidates():
    mlp = joined(R/'price_mlp_retry1/component_probabilities.parquet', R/'price_mlp_holdout/component_probabilities.parquet')
    blend = joined(R/'price_mlp_retry1/selected_blend_predictions.parquet', R/'price_mlp_holdout/selected_blend_predictions.parquet')
    return {
        'P0': joined(R/'price_v2/P0_predictions.parquet', R/'price_v2_holdout_P3/P0_predictions.parquet'),
        'P3': joined(R/'price_v2/P3_predictions.parquet', R/'price_v2_holdout_P3/P3_predictions.parquet'),
        'ensemble': joined(R/'price_v2/ensemble_predictions.parquet', R/'price_v2_holdout_ensemble/ensemble_predictions.parquet'),
        'stacked': joined(R/'price_v2/stacked_predictions.parquet', R/'price_v2_holdout_stacked/stacked_predictions.parquet'),
        'lgb_mlp_blend': blend.assign(probability=.75*mlp.lightgbm_probability + .25*mlp.mlp_probability),
        'mlp': mlp.rename(columns={'mlp_probability': 'probability'}).assign(prediction=np.nan),
    }


def ranking_vs_cutoff(models):
    rows = []
    for name, p in models.items():
        years = market_days(p.index).year
        for year in sorted(set(years)):
            m = years == year
            a, pr = p.actual[m].to_numpy(), p.probability[m].to_numpy()
            best, cut = oracle(a, pr)
            realised = p.prediction[m].to_numpy()
            has = np.isfinite(realised).all()
            rows.append(dict(model=name, year=year, prevalence_1=100*a.mean(), AUC=100*roc_auc_score(a, pr),
                             oracle_BA=best, oracle_cutoff=cut, BA_at_057=balanced_accuracy(a, pr > THRESHOLD),
                             realised_BA=balanced_accuracy(a, realised) if has else np.nan,
                             recall_0=100*np.mean(realised[a == 0] == 0) if has else np.nan,
                             recall_1=100*np.mean(realised[a == 1] == 1) if has else np.nan,
                             realised_cutoffs=';'.join(f'{t:.2f}' for t in sorted(p.threshold[m].unique()))
                             if 'threshold' in p else ''))
    return pd.DataFrame(rows)


def count_shape(p0, targets):
    a = p0.actual.to_numpy().reshape(-1, 48)
    pr = p0.probability.to_numpy().reshape(-1, 48)
    pred = p0.prediction.to_numpy().reshape(-1, 48)
    price = targets.price.loc[p0.index].to_numpy().reshape(-1, 48)
    years = market_days(p0.index).year.to_numpy().reshape(-1, 48)[:, 0]
    k, kp, ks = a.sum(1), pred.sum(1), pr.sum(1)
    model_rank = np.argsort(np.argsort(-pr, axis=1), axis=1)
    price_rank = np.argsort(np.argsort(-price, axis=1), axis=1)
    oracle_count = model_rank < k[:, None]           # true count, P0 within-day ranking
    oracle_shape = price_rank < kp[:, None]          # true ranking, P0 count
    soft_topk = model_rank < np.round(ks)[:, None]   # P0's own expected count
    yesterday = targets.yesterday.loc[p0.index].to_numpy().reshape(-1, 48).sum(1)
    rows = []
    for year in sorted(set(years)):
        m = years == year
        rows.append(dict(year=year, P0_BA=balanced_accuracy(a[m], pred[m]),
                         oracle_count_BA=balanced_accuracy(a[m], oracle_count[m]),
                         oracle_shape_BA=balanced_accuracy(a[m], oracle_shape[m]),
                         soft_topk_BA=balanced_accuracy(a[m], soft_topk[m]),
                         mean_true_count=k[m].mean(), P0_count_MAE=np.abs(k-kp)[m].mean(),
                         P0_count_bias=(kp-k)[m].mean(), P0_count_corr=np.corrcoef(k[m], kp[m])[0, 1],
                         yesterday_count_MAE=np.abs(k-yesterday)[m].mean()))
    return pd.DataFrame(rows)


PERIODS = [('2015-2022', 2015, 2022), ('2023-2024', 2023, 2024), ('2025', 2025, 2025), ('2026', 2026, 2026)]


def grouped_errors(p0, targets):
    days = market_days(p0.index)
    years = days.year
    error = (p0.actual != p0.prediction).to_numpy()
    start = p0.index - pd.Timedelta(minutes=30)
    slot = np.asarray(start.hour*2 + start.minute//30)
    block = pd.cut(slot, [-1, 11, 17, 31, 41, 47], labels=['00-06', '06-09', '09-16', '16-21', '21-24'])
    fraction = p0.actual.groupby(days).transform('mean').to_numpy()
    fraction_bin = pd.cut(fraction, [-.01, .05, .25, .5, .75, .95, 1.], labels=['<=5%', '5-25%', '25-50%', '50-75%', '75-95%', '>95%'])
    prices = targets.price
    q = prices.groupby(market_days(prices.index)).quantile([.25, .75]).unstack()
    iqr = (q[.75] - q[.25]).clip(lower=1).shift(1)
    z = ((targets.price - targets.tau).loc[p0.index] / pd.Series(days, index=p0.index).map(iqr)).abs()
    margin_bin = pd.cut(z.to_numpy(), [0, .05, .1, .25, .5, 1, 2, np.inf], include_lowest=True).astype(str)
    out = []
    for kind, groups in [('slot_block', np.asarray(block)), ('day_class1_fraction', np.asarray(fraction_bin)),
                         ('abs_margin_over_prev_IQR', margin_bin)]:
        for label, low, high in PERIODS:
            m = (years >= low) & (years <= high)
            t = pd.DataFrame(dict(g=groups[m], e=error[m], a=p0.actual.to_numpy()[m], p=p0.prediction.to_numpy()[m]))
            total = t.e.sum()
            for g, part in t.groupby('g', observed=True):
                out.append(dict(kind=kind, period=label, group=g, rows=len(part), row_share=100*len(part)/len(t),
                                error_rate=100*part.e.mean(), error_share=100*part.e.sum()/total,
                                actual_rate_1=100*part.a.mean(), predicted_rate_1=100*part.p.mean()))
    return pd.DataFrame(out)


def monthly(p0, targets):
    t = targets.loc[p0.index]
    frame = pd.DataFrame(dict(a=p0.actual, p=p0.prediction, b=t.yesterday, w=t.last_week, price=t.price), index=p0.index)
    days = market_days(frame.index)
    rows = []
    for month, g in frame.groupby(days.to_period('M')):
        rows.append(dict(month=str(month), BA=balanced_accuracy(g.a, g.p), yesterday_BA=balanced_accuracy(g.a, g.b),
                         last_week_BA=balanced_accuracy(g.a, g.w), prevalence_1=100*g.a.mean(),
                         recall_0=100*np.mean(g.p[g.a == 0] == 0), recall_1=100*np.mean(g.p[g.a == 1] == 1),
                         negative_price_share=100*(g.price < 0).mean(), mean_price=g.price.mean(),
                         daily_mean_sd=g.price.groupby(market_days(g.index)).mean().std()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=R/'price_v3_diagnostics')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from model import load_data
    targets = harness_targets(load_data(DATA))
    models = candidates()
    for name, p in models.items():
        if not (targets.actual.loc[p.index] == p.actual).all():
            raise ValueError(f'{name}: saved labels differ from harness targets')
        if len(p) % 48:
            raise ValueError(f'{name}: incomplete market days')
    p0 = models['P0']
    ranking = ranking_vs_cutoff(models)
    ranking.to_csv(args.output/'ranking_vs_cutoff.csv', index=False)
    decomposition = count_shape(p0, targets)
    decomposition.to_csv(args.output/'count_shape_oracle.csv', index=False)
    grouped_errors(p0, targets).to_csv(args.output/'p0_grouped_errors.csv', index=False)
    monthly(p0, targets).to_csv(args.output/'p0_monthly.csv', index=False)
    t = targets.loc[p0.index]
    weeks = weekly_gain(p0.actual, p0.prediction, t.yesterday)
    weeks.to_csv(args.output/'p0_weekly_gain.csv', index=False)
    year = pd.to_datetime(weeks.start).dt.year
    summary = weeks.groupby(year).G.agg(weeks='size', median='median', p10=lambda s: s.quantile(.1),
                                        p90=lambda s: s.quantile(.9), share_ge_8=lambda s: 100*(s >= 8).mean())
    summary.to_csv(args.output/'p0_weekly_gain_by_year.csv')
    baselines = []
    for y in sorted(set(market_days(p0.index).year)):
        m = market_days(p0.index).year == y
        baselines.append(dict(year=y, P0_BA=balanced_accuracy(p0.actual[m], p0.prediction[m]),
                              yesterday_BA=balanced_accuracy(t.actual[m], t.yesterday[m]),
                              last_week_BA=balanced_accuracy(t.actual[m], t.last_week[m])))
    pd.DataFrame(baselines).assign(gain_over_yesterday=lambda d: d.P0_BA - d.yesterday_BA).to_csv(
        args.output/'p0_vs_baselines_by_year.csv', index=False)
    pd.set_option('display.width', 200)
    print(ranking.pivot(index='year', columns='model', values='oracle_BA').round(2))
    print(decomposition.round(2).to_string(index=False))
    print(summary.round(2))


if __name__ == '__main__':
    main()
