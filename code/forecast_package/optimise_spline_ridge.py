"""Chronological demand-only spline experiments; original artifacts are preserved."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from spline_ridge import SplineRidgeDemand
from model import load_data, training_table, market_days, solar_trend_features


class OptimisedSplineDemand(SplineRidgeDemand):
    def __init__(self, alpha=10.0, n_knots=6, features='original'):
        super().__init__(alpha, n_knots)
        self.features = features

    def _design(self, features, fit=False):
        base = super()._design(features, fit)
        if self.features == 'original':
            return base
        x = solar_trend_features(features)
        columns = ['solar_delta_anchor', 'trend_years', 'solar_delta_trend']
        extra = x[columns].copy()
        extra['temp_delta_anchor'] = (x.syd_temp_change_1d + x.syd_temp_change_7d) / 2
        extra['recent_shape'] = x.nsw_demand_lag_1 - x.nsw_demand_weekly_mean
        extra['anchor_change'] = x.nsw_demand_lag_1 - x.demand_anchor
        if self.features == 'rich':
            for c in x:
                if c.startswith(('vic_demand_', 'qld_demand_', 'mel_', 'bne_')):
                    extra[c] = x[c]
        if fit:
            self.extra_scaler_ = StandardScaler()
        method = 'fit_transform' if fit else 'transform'
        values = getattr(self.extra_scaler_, method)(extra).astype(np.float32)
        slot = self.slot_.transform(x[['slot']]).tocsr()
        blocks = [base, sparse.csr_matrix(values)]
        # Changes relative to the anchor can have different effects at each hour.
        blocks.extend(slot.multiply(values[:, [j]]) for j in (0, 2, 3, 4, 5))
        if self.features == 'rich':
            blocks.extend(slot.multiply(x[f'dow_{j}'].to_numpy()[:, None]) for j in range(7))
        design = sparse.hstack(blocks, format='csr', dtype=np.float32)
        if fit:
            self.extra_design_scaler_ = StandardScaler(with_mean=False)
        return getattr(self.extra_design_scaler_, method)(design)


def report(prediction, name):
    days = market_days(prediction.index)
    slot = (prediction.index - pd.Timedelta(minutes=30)).hour
    midday = (slot >= 10) & (slot < 15)
    rows = []
    periods = [('all', np.ones(len(days), dtype=bool))]
    periods += [(str(y), days.year == y) for y in sorted(set(days.year))]
    for period, mask in periods:
        for group, subset in [('all', mask), ('midday', mask & midday), ('other', mask & ~midday)]:
            p = prediction.loc[subset]
            if p.empty:
                continue
            error = p.prediction - p.actual
            rows.append(dict(candidate=name, period=period, group=group, rows=len(p),
                             MAPE=float((error.abs() / p.actual).mean() * 100),
                             MAE=float(error.abs().mean()), bias_mw=float(error.mean())))
    return rows


def backtest(x, y, config, start, stop, name, output):
    days = market_days(x.index)
    predictions, fits = [], []
    for year in range(start, stop + 1):
        effective = year if config['schedule'] == 'annual' else (2015 if year == 2015 else year - year % 2)
        if year == start or effective != previous:
            train = days.year < effective
            if config['window']:
                train &= days.year >= effective - config['window']
            model = OptimisedSplineDemand(**{k: config[k] for k in ('alpha', 'n_knots', 'features')})
            model.fit(x.loc[train], y.loc[train])
            fits.append(dict(candidate=name, effective_year=effective, training_start=model.training_start_,
                             training_end=model.training_end_, rows=int(train.sum()), seconds=model.fit_seconds_))
        score = days.year == year
        predictions.append(pd.DataFrame({'actual': y.loc[score, 'demand'],
                                         'prediction': model.predict(x.loc[score])}))
        previous = effective
    p = pd.concat(predictions)
    p.to_parquet(output / f'{name}_{start}_{stop}.parquet')
    return report(p, name), fits


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(root / 'data/nem_weather_halfhourly_2010_2026.csv'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-selection', type=Path, help='Skip experiments and fit the selected configuration')
    parser.add_argument('--cutoff', default='2024-12-31', help='Final fit cutoff; later data require an explicit override')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Choose an empty output directory')
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        frame = load_data(args.data)
        cutoff = pd.Timestamp(args.cutoff) if args.train_selection else pd.Timestamp('2024-12-31')
        frame = frame.loc[market_days(frame.index) <= cutoff]
        x, y = training_table(frame)
        if args.train_selection:
            selected = json.loads(args.train_selection.read_text())
        else:
            if not set(range(2010, 2025)).issubset(set(market_days(x.index).year)):
                parser.error('Experiments require complete 2010-2024 history')
            configs = []
            for window, schedule, features in [(0, 'biennial', 'original'), (2, 'biennial', 'original'),
                     (4, 'biennial', 'original'), (2, 'annual', 'original'), (4, 'annual', 'original'),
                     (2, 'annual', 'trend'), (2, 'annual', 'rich'), (4, 'annual', 'rich')]:
                configs.append(dict(window=window, schedule=schedule, features=features, alpha=10., n_knots=6))
            all_rows, all_fits, scores = [], [], []
            def evaluate(config, name):
                rows, fits = backtest(x, y, config, 2015, 2022, name, args.output)
                score = next(r['MAPE'] for r in rows if r['period'] == 'all' and r['group'] == 'all')
                all_rows.extend(rows)
                all_fits.extend(fits)
                scores.append(dict(name=name, config=config, MAPE=score))
                (args.output / 'candidates.json').write_text(json.dumps(scores, indent=2) + '\n')
                pd.DataFrame(all_rows).to_csv(args.output / 'development_metrics.csv', index=False)
                pd.DataFrame(all_fits).to_csv(args.output / 'fits.csv', index=False)
                print(f'{name}: MAPE={score:.4f}%; elapsed={time.perf_counter()-started:.1f}s', flush=True)
            for i, config in enumerate(configs):
                evaluate(config, f'structure_{i}')
                # Stop adding candidates well before the two-hour target.
                if time.perf_counter() - started > 3600:
                    break
            winner = min(scores, key=lambda r: r['MAPE'])
            for alpha, knots in [(100., 6), (10., 4)]:
                if time.perf_counter() - started < 4500:
                    evaluate(dict(winner['config'], alpha=alpha, n_knots=knots), f'tune_{alpha}_{knots}')
            selected = min(scores, key=lambda r: r['MAPE'])
            (args.output / 'selection.json').write_text(json.dumps(selected, indent=2) + '\n')
            # Selection is fixed before inspecting 2023-2024 results.
            confirmation, fits = [], []
            for name, config in [('original', configs[0]), ('selected', selected['config'])]:
                rows, records = backtest(x, y, config, 2023, 2024, name, args.output)
                confirmation.extend(rows)
                fits.extend(records)
            pd.DataFrame(confirmation).to_csv(args.output / 'confirmation_metrics.csv', index=False)
            pd.DataFrame(fits).to_csv(args.output / 'confirmation_fits.csv', index=False)
            reference_report(root, args.output)
        config = selected['config']
        train = np.ones(len(x), dtype=bool)
        if config['window']:
            end = market_days(x.index).max()
            train = market_days(x.index) >= end + pd.Timedelta(days=1) - pd.DateOffset(years=config['window'])
        model = OptimisedSplineDemand(**{k: config[k] for k in ('alpha', 'n_knots', 'features')})
        model.fit(x.loc[train], y.loc[train])
        joblib.dump(model, args.output / 'demand.joblib', compress=3)
        manifest = dict(selection=selected, training_start=model.training_start_, training_end=model.training_end_,
                        total_seconds=time.perf_counter()-started, cutoff=str(cutoff),
                        confirmation='Exploratory: 2023-2024 were previously inspected', threads=2)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest, indent=2), flush=True)


def reference_report(root, output):
    """Compare saved LightGBM predictions on exactly the baseline scoring dates."""
    path = root / 'results/half_hour_predictions.parquet'
    if not path.exists():
        return
    rows = []
    saved = pd.read_parquet(path, columns=['nsw_demand', 'adapted_periodic_expanding_demand'])
    for phase, file in [('development', 'structure_0_2015_2022.parquet'),
                        ('confirmation', 'original_2023_2024.parquet')]:
        expected = pd.read_parquet(output / file)
        aligned = saved.loc[expected.index]
        np.testing.assert_allclose(aligned.nsw_demand, expected.actual)
        p = pd.DataFrame({'actual': aligned.nsw_demand,
                          'prediction': aligned.adapted_periodic_expanding_demand})
        rows.extend(dict(r, phase=phase) for r in report(p, 'saved_lightgbm'))
    pd.DataFrame(rows).to_csv(output / 'lightgbm_reference_metrics.csv', index=False)


if __name__ == '__main__':
    from optimise_spline_ridge import main as run
    run()
