"""WP2: daily class-1 count ("level") model.

Target for forecast day D+1: the fraction of its 48 intervals priced above the
previous day's mean (k/48). Inputs are only the day-constant columns of the
causal price features (yesterday/last-week profiles relative to the threshold,
previous-day statistics, daily next-day weather, calendar, daily price context),
so every input is available at the end of day D. Configuration and decoder
weight are chosen on 2014 after fitting through 2013; out-of-fold predictions
follow the same replacement boundaries as code/price_v2/stack.py.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from common import ROOT, DATA, THRESHOLD, market_days, balanced_accuracy

CONFIGS = {
    'A': dict(n_estimators=300, num_leaves=15, learning_rate=.03, min_child_samples=40, colsample_bytree=.5, reg_lambda=5.),
    'B': dict(n_estimators=600, num_leaves=7, learning_rate=.02, min_child_samples=60, colsample_bytree=.3, reg_lambda=5.),
    'C': dict(n_estimators=300, num_leaves=31, learning_rate=.03, min_child_samples=20, colsample_bytree=.5, reg_lambda=5.),
    'D': dict(n_estimators=200, num_leaves=15, learning_rate=.05, min_child_samples=80, colsample_bytree=.8, reg_lambda=10.),
}
ALPHAS = (0., .25, .5, .75, 1.)
# Fold starts: annual to 2016, then the biennial price replacement schedule.
BOUNDARIES = pd.to_datetime([f'{y}-01-01' for y in (2011, 2012, 2013, 2014, 2015, 2016, 2018, 2020, 2022, 2024, 2026, 2027)])
PC_DAILY = ['pc_skew', 'pc_tail', 'pc_std_ratio', 'pc_spike_fraction', 'pc_negative_fraction', 'pc_exceed_all_28',
            'pc_late_change', 'pc_mean_momentum', 'pc_weekly_anomaly']


def day_features(rows, columns):
    """One row per market day from any row-level frame (batch or the 48 live rows)."""
    days = market_days(rows.index)
    first = rows.groupby(days, sort=True).head(1)
    out = first.loc[:, columns].copy()
    for name, prefix in [('lvl_frac_above_lag_1', 'nsw_price_profile_lag_1_margin_slot_'),
                         ('lvl_frac_above_lag_7', 'nsw_price_profile_lag_7_margin_slot_'),
                         ('lvl_frac_above_weekly', 'nsw_price_profile_weekly_margin_slot_')]:
        margins = first[[f'{prefix}{s}' for s in range(48)]].to_numpy()
        out[name] = (margins > 0).mean(axis=1)
        out[name.replace('frac_above', 'median_margin')] = np.median(margins, axis=1)
    out.index = market_days(first.index)
    return out.astype(np.float32)


def constant_columns(x, before='2014-01-01'):
    """Columns with zero within-day spread on the training prefix only."""
    days = market_days(x.index)
    prefix = x.loc[days < pd.Timestamp(before)].to_numpy(dtype=np.float32)
    cube = prefix.reshape(-1, 48, prefix.shape[1])
    spread = np.nanmax(cube, axis=1) - np.nanmin(cube, axis=1)
    return [c for c, s in zip(x.columns, np.nanmax(spread, axis=0)) if s == 0]


def make_model(config, seed=9417):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective='cross_entropy', subsample=.8, subsample_freq=1, max_bin=127, n_jobs=2,
                         verbosity=-1, deterministic=True, force_col_wise=True, random_state=seed, **config)


def fit_level(table, config, cutoff, seed=9417):
    train = table.index < cutoff
    if not train.any():
        raise ValueError('No level-model history before the cutoff')
    model = make_model(config, seed).fit(table.loc[train].drop(columns=['target']), table.loc[train, 'target'])
    model.training_end_ = str(table.index[train].max().date())
    return model


def predict_level(model, table):
    x = table.drop(columns=['target'], errors='ignore')
    if (x.index <= pd.Timestamp(model.training_end_)).any():
        raise ValueError('Level predictions must follow the level model training end')
    return np.clip(model.predict(x), 1e-4, 1 - 1e-4)


def build_table(cache):
    """Day table from the WP1 memory-mapped 571-feature cache plus daily price context."""
    from model import load_data
    from price_context import price_context_frame
    x = np.load(cache/'x.npy', mmap_mode='r')
    index = pd.DatetimeIndex(np.load(cache/'index.npy'))
    columns = json.loads((cache/'columns.json').read_text())
    price = np.load(cache/'price.npy')
    if len(index) % 48 or (market_days(index[::48]) + pd.Timedelta(minutes=30) != index[::48]).any():
        raise ValueError('Cache must hold complete market days')
    prefix = int((market_days(index) < pd.Timestamp('2014-01-01')).sum())
    spread = np.empty(len(columns))
    for j in range(0, len(columns), 32):   # chunked to keep memory small beside running fits
        cube = np.asarray(x[:prefix, j:j+32], dtype=np.float32).reshape(-1, 48, min(32, len(columns)-j))
        spread[j:j+32] = np.nanmax(np.nanmax(cube, axis=1) - np.nanmin(cube, axis=1), axis=0)
    keep = [c for c, s in zip(columns, spread) if s == 0]
    first = pd.DataFrame(np.array(x[::48]), index=index[::48], columns=columns).loc[:, keep]
    context = price_context_frame(load_data(DATA), 'both').loc[index, PC_DAILY]
    daily_context = constant_columns(context)
    rows = pd.concat([first, context.iloc[::48][daily_context]], axis=1)
    keep = keep + daily_context
    table = day_features(rows, keep)
    table['target'] = price.reshape(-1, 48).mean(axis=1)
    return table, keep


def oof(table, config):
    out = pd.Series(np.nan, index=table.index)
    audit = []
    for begin, end in zip(BOUNDARIES[:-1], BOUNDARIES[1:]):
        score = (table.index >= begin) & (table.index < end)
        if not score.any():
            continue
        model = fit_level(table, config, begin)
        out[score] = predict_level(model, table.loc[score])
        audit.append(dict(fold_start=str(begin.date()), training_end=model.training_end_,
                          scored_days=int(score.sum())))
    return out, pd.DataFrame(audit)


def decode(probability, k_hat, alpha):
    """Top-k per day; alpha=0 reproduces probability > 0.57 exactly."""
    p = np.asarray(probability).reshape(-1, 48)
    k_model = (p > THRESHOLD).sum(axis=1)
    k = np.rint(alpha * 48 * np.asarray(k_hat) + (1 - alpha) * k_model).astype(int)
    rank = np.argsort(np.argsort(-p, axis=1, kind='stable'), axis=1, kind='stable')
    return (rank < k[:, None]).astype(int).ravel()


def row_features(k_hat, index):
    """Broadcast daily predictions to the 48 rows: the stacked-arm extra columns."""
    daily = pd.Series(np.asarray(k_hat, dtype=float), index=pd.DatetimeIndex(k_hat.index))
    days = market_days(index)
    frac = daily.reindex(days).to_numpy()
    return pd.DataFrame(dict(lvl_frac_hat=frac, lvl_logit_hat=np.log(frac/(1-frac))), index=index).astype(np.float32)


def screen(output, table, k_oof):
    """Decoder screen on SAVED P0 probabilities; no interval refit."""
    from research import paired
    R = ROOT/'results'
    p0 = pd.concat([pd.read_parquet(R/'price_v2/P0_predictions.parquet'),
                    pd.read_parquet(R/'price_v2_holdout_P3/P0_predictions.parquet')])
    p0 = p0[~p0.index.duplicated()].sort_index()
    selection = json.loads((output/'selection.json').read_text())
    alpha = selection['alpha']
    days = market_days(p0.index).unique()
    k_hat = k_oof.loc[days]
    if k_hat.isna().any():
        raise ValueError('Missing out-of-fold level predictions')
    candidate = p0.copy()
    candidate['prediction'] = decode(p0.probability, k_hat, alpha)
    if not (decode(p0.probability, k_hat, 0.) == p0.prediction.to_numpy()).all():
        raise AssertionError('alpha=0 must reproduce P0')
    truth = p0.actual.to_numpy().reshape(-1, 48).sum(axis=1)
    rows = []
    years = pd.DatetimeIndex(days).year
    for year in sorted(set(years)):
        m = years == year
        rm = market_days(p0.index).year == year
        rows.append(dict(year=year, P0_BA=balanced_accuracy(p0.actual[rm], p0.prediction[rm]),
                         screen_BA=balanced_accuracy(candidate.actual[rm], candidate.prediction[rm]),
                         P0_count_MAE=np.abs(p0.prediction.to_numpy().reshape(-1, 48).sum(1) - truth)[m].mean(),
                         level_count_MAE=np.abs(48*k_hat.to_numpy() - truth)[m].mean(),
                         level_count_bias=(48*k_hat.to_numpy() - truth)[m].mean(),
                         decoded_count_MAE=np.abs(candidate.prediction.to_numpy().reshape(-1, 48).sum(1) - truth)[m].mean()))
    by_year = pd.DataFrame(rows).assign(delta=lambda d: d.screen_BA - d.P0_BA)
    by_year.to_csv(output/'screen_by_year.csv', index=False)
    d = market_days(p0.index).year
    dev, conf, late = d <= 2022, (d >= 2023) & (d <= 2024), d >= 2025
    deltas = {label: paired(p0.loc[m], candidate.loc[m]) for label, m in
              [('development_2015_2022', dev), ('confirmation_2023_2024', conf), ('reused_2025_2026', late)]}
    accuracy = 100*(np.mean(candidate.actual[dev] == candidate.prediction[dev]) - np.mean(p0.actual[dev] == p0.prediction[dev]))
    positive = int((by_year.loc[by_year.year <= 2022, 'delta'] > 0).sum())
    gate = dict(alpha=alpha, **{f'{k}_{s}': v for k, d_ in deltas.items() for s, v in d_.items()},
                positive_development_years=positive, development_accuracy_delta_pp=accuracy,
                passes_development_gate=bool(deltas['development_2015_2022']['delta_BA_pp'] >= .25
                                             and deltas['development_2015_2022']['low'] > 0 and positive >= 6
                                             and deltas['confirmation_2023_2024']['delta_BA_pp'] > 0 and accuracy >= -.25))
    (output/'screen_gate.json').write_text(json.dumps(gate, indent=2))
    candidate.to_parquet(output/'screen_predictions.parquet')
    print(by_year.round(3).to_string(index=False))
    print(json.dumps(gate, indent=2))


def select(output, table):
    """Stage-1 discipline: fit through 2013, choose config by 2014 count MAE, then alpha by 2014 BA."""
    valid = (table.index >= pd.Timestamp('2014-01-01')) & (table.index < pd.Timestamp('2015-01-01'))
    scores = {}
    for name, config in CONFIGS.items():
        model = fit_level(table, config, pd.Timestamp('2014-01-01'))
        scores[name] = float(np.mean(np.abs(48*(predict_level(model, table.loc[valid]) - table.loc[valid, 'target']))))
    best = min(scores, key=lambda n: (scores[n], n))
    model = fit_level(table, CONFIGS[best], pd.Timestamp('2014-01-01'))
    k_hat = pd.Series(predict_level(model, table.loc[valid]), index=table.index[valid])
    validation = pd.read_parquet(ROOT/'results/price_v2/P0_validation.parquet')
    if not (market_days(validation.index).unique() == k_hat.index).all():
        raise ValueError('2014 validation days differ')
    alphas = {a: balanced_accuracy(validation.actual, decode(validation.probability, k_hat, a)) for a in ALPHAS}
    alpha = max(alphas, key=lambda a: (round(alphas[a], 10), -a))   # ties favour the simpler P0 decoder
    selection = dict(config_name=best, config=CONFIGS[best], validation_count_MAE=scores,
                     validation_BA_by_alpha={str(a): s for a, s in alphas.items()}, alpha=alpha,
                     selected_on='fit through 2013, score 2014', registered=time.strftime('%Y-%m-%d %H:%M:%S'))
    (output/'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))
    return selection


def harness_extras(output, table, k_oof, config):
    """Per-cutoff row features for the WP1 stacked arm: OOF before the cutoff, a fresh fit after."""
    from live_horizon import CUTOFFS, window
    folder = output/'harness_extra'
    folder.mkdir(exist_ok=True)
    index = pd.DatetimeIndex(np.load(Path(output/'cache_index.npy')))
    for cutoff in CUTOFFS:
        model = fit_level(table, config, cutoff)
        _, hi = window(cutoff)
        values = k_oof.copy()
        score = (table.index >= cutoff) & (table.index < hi)
        values[score] = predict_level(model, table.loc[score])
        values = values[(values.index >= pd.Timestamp('2011-01-01')) & (values.index < hi)]
        rows = index[(market_days(index) >= values.index.min()) & (market_days(index) <= values.index.max())]
        row_features(values, rows).to_parquet(folder/f'{cutoff.date()}.parquet')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'results'/'price_v3_level')
    ap.add_argument('--cache', type=Path, default=Path(tempfile.gettempdir())/'price_v3_cache')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        table, keep = build_table(args.cache)
        (args.output/'day_columns.json').write_text(json.dumps(keep, indent=2))
        print(f'Day table: {table.shape}, {len(keep)} day-constant inputs', flush=True)
        selection = (json.loads((args.output/'selection.json').read_text()) if (args.output/'selection.json').exists()
                     else select(args.output, table))
        k_oof, audit = oof(table, selection['config'])
        audit.to_csv(args.output/'oof_audit.csv', index=False)
        k_oof.rename('k_hat').to_frame().to_parquet(args.output/'oof_daily.parquet')
        np.save(args.output/'cache_index.npy', np.load(args.cache/'index.npy'))
        row_features(k_oof.dropna(), pd.DatetimeIndex(np.load(args.cache/'index.npy'))[
            market_days(pd.DatetimeIndex(np.load(args.cache/'index.npy'))) >= pd.Timestamp('2011-01-01')]
        ).to_parquet(args.output/'oof_rows.parquet')
        screen(args.output, table, k_oof)
        harness_extras(args.output, table, k_oof, selection['config'])


if __name__ == '__main__':
    main()
