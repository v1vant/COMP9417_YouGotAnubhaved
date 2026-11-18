"""Run: python -m unittest discover -s code/price_v3 -p test_v3.py -v"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import common  # noqa: F401  (adds code/price_v2 to sys.path)
from model import MARKET, WEATHER
from pipeline import table
from common import market_days, THRESHOLD


def synthetic(days=150, start='2013-12-01 00:30'):
    idx = pd.date_range(start, periods=days*48, freq='30min')
    a = np.arange(len(idx))
    rng = np.random.default_rng(0)
    f = pd.DataFrame(index=idx)
    for c in MARKET:
        f[c] = (5000 + np.sin(a/20)*100 if c.endswith('demand') else 40 + np.sin(a/10)*20 + rng.normal(0, 5, len(a)))
    for c in WEATHER:
        f[c] = 20 + np.cos(a/40)
    return f


def write_cache(folder, frame):
    x, y = table(frame)
    np.save(folder/'x.npy', x.to_numpy(dtype=np.float32))
    np.save(folder/'index.npy', x.index.to_numpy())
    np.save(folder/'price.npy', y.price.to_numpy(dtype=np.int8))
    (folder/'columns.json').write_text(json.dumps(list(x.columns)))
    return x, y


class LiveHorizonTests(unittest.TestCase):
    def test_training_stops_before_cutoff_and_ignores_later_outcomes(self):
        import live_horizon as lh
        cutoff = pd.Timestamp('2014-02-01')
        frame = synthetic()
        changed = frame.copy()
        later = market_days(changed.index) >= cutoff + pd.Timedelta(days=1)
        changed.loc[later, 'nsw_price'] = 1e4
        results = []
        for f in (frame, changed):
            with tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                write_cache(folder, f)
                lh._init(folder)
                record = lh.run_one('t', dict(lh.VARIANTS['V0'], trees=5), cutoff, folder/'p.parquet')
                results.append((record, pd.read_parquet(folder/'p.parquet')))
        for record, _ in results:
            self.assertLess(pd.Timestamp(record['training_end']), cutoff)
        first = market_days(results[0][1].index) == cutoff
        np.testing.assert_array_equal(results[0][1].probability[first], results[1][1].probability[first])
        self.assertEqual(market_days(results[0][1].index).min(), cutoff)
        self.assertEqual(market_days(results[0][1].index).max(), pd.Timestamp('2014-03-31'))

    def test_cutoffs_tile_the_reused_period(self):
        import live_horizon as lh
        self.assertEqual(len(lh.CUTOFFS), 18)
        for a, b in zip(lh.CUTOFFS, lh.CUTOFFS[1:]):
            self.assertEqual(lh.window(a)[1], b)
        self.assertEqual(lh.window(lh.CUTOFFS[-1])[1], pd.Timestamp('2026-11-01'))


class LevelTests(unittest.TestCase):
    def test_alpha_zero_decoder_reproduces_threshold(self):
        from level import decode
        rng = np.random.default_rng(1)
        p = rng.random(48*30)
        k_hat = rng.random(30)
        np.testing.assert_array_equal(decode(p, k_hat, 0.), (p > THRESHOLD).astype(int))
        full = decode(p, k_hat, 1.).reshape(-1, 48).sum(axis=1)
        np.testing.assert_array_equal(full, np.rint(48*k_hat).astype(int))

    def test_day_table_live_batch_parity_and_causality(self):
        from level import day_features
        from model import forecast_features
        frame = synthetic(70)
        x, _ = table(frame)
        columns = [c for c in x.columns if c.startswith(('nsw_price_profile_', 'nsw_price_previous_', 'syd_temp_today'))]
        batch = day_features(x, columns)
        day = market_days(frame.index[-48:])[0]
        history, weather = frame.iloc[:-48], frame.iloc[-48:][WEATHER]
        live = day_features(forecast_features(history, weather), columns)
        pd.testing.assert_frame_equal(batch.loc[[day]], live)
        changed = frame.copy()
        changed.loc[changed.index[-48:], MARKET] = -1e9
        altered = day_features(table(changed)[0], columns)
        pd.testing.assert_frame_equal(batch.loc[[day]], altered.loc[[day]])

    def test_level_fits_respect_cutoffs(self):
        from level import fit_level, predict_level, CONFIGS
        days = pd.date_range('2012-01-01', periods=900)
        rng = np.random.default_rng(2)
        t = pd.DataFrame(dict(a=rng.random(900), target=rng.random(900)), index=days)
        cutoff = pd.Timestamp('2013-06-01')
        model = fit_level(t, dict(CONFIGS['D'], n_estimators=5), cutoff)
        self.assertLess(pd.Timestamp(model.training_end_), cutoff)
        with self.assertRaises(ValueError):
            predict_level(model, t.loc[t.index >= cutoff - pd.Timedelta(days=1)])
        self.assertTrue(np.isfinite(predict_level(model, t.loc[t.index >= cutoff])).all())


if __name__ == '__main__':
    unittest.main()


PACKAGE_SCRIPT = r"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from train import price_table, fit_members
from recipe import fit_price, weather_shape_names, abs_margin
from model import Forecaster, price_probability, WEATHER
import base_model
frame = pd.read_parquet(sys.argv[1])
x, y = price_table(frame)
xs, _ = price_table(frame, weather_shape=True)
x.to_parquet('x.parquet'); y.to_parquet('y.parquet'); xs.to_parquet('xs.parquet')
cut = x.index[-48*20 - 1]   # last interval of a complete market day
out = {}
for hl in (None, 2.):
    m = fit_price(x.loc[:cut], y.price.loc[:cut], 9418, dict(n_estimators=5), hl)
    out[str(hl)] = m.predict_proba(x.iloc[-48:])[:, 1].tolist()
cand = dict(seeds=[9418, 9419], threshold=.5, weather_shape=True, near_threshold_scale=.1,
            params=dict(n_estimators=5, num_leaves=7, learning_rate=.05, colsample_bytree=.5, min_child_samples=40))
cx, members = fit_members(frame, cand)
out['package_ws'] = np.mean([m.predict_proba(cx.iloc[-48:])[:, 1] for m in members], axis=0).tolist()
out['package_ws_columns'] = list(cx.columns)
json.dump(out, open('probs.json', 'w'))
class Const:
    def __init__(self, v): self.v = v; self.seen = None
    def get_params(self): return {}
    def predict(self, z): return np.full(len(z), 100.)
    def predict_proba(self, z):
        self.seen = list(z.columns); return np.tile([1-self.v, self.v], (len(z), 1))
members = [Const(.5), Const(.6), Const(.67)]
end = str(base_model.market_days(x.index[-49:-48])[0])
models = dict(demand=dict(task='demand', estimator=Const(0), features=list(x.columns)),
              price=dict(task='price', estimator=members[0], members=members, features=list(x.columns),
                         threshold=.58, training_end=end, config={}))
a = Forecaster.__new__(Forecaster); a.models = models; a.demand_mlp = None
r = base_model.Forecaster.__new__(base_model.Forecaster); r.models = models; r.demand_mlp = None
history, weather = frame.iloc[:-48], frame.iloc[-48:][WEATHER]
pa, pr = a.predict(history, weather), r.predict(history, weather)
assert (pa.nsw_demand_mw.to_numpy() == pr.nsw_demand_mw.to_numpy()).all(), 'demand differs'
assert (pa.nsw_price_up == 1).all(), 'bag mean .59 must exceed .58'
assert (pr.nsw_price_up == 0).all(), 'reference uses only the first member (.5 < .58)'
ws_members = [Const(.6)]
a.models = dict(models, price=dict(models['price'], members=ws_members, estimator=ws_members[0],
                                   features=list(xs.columns), config={'weather_shape': True}))
pw = a.predict(history, weather)
assert ws_members[0].seen == list(xs.columns) and len(xs.columns) == x.shape[1] + 26, 'weather-shape inputs'
assert (pw.nsw_demand_mw.to_numpy() == pr.nsw_demand_mw.to_numpy()).all(), 'demand changed with weather shape'
late = dict(models, price=dict(models['price'], training_end=str(frame.index[-1])))
a.models = late
a.predict(history, weather)   # in-sample days must still predict (check_submission uses them)
print('PACKAGE_OK')
"""


class PackageTests(unittest.TestCase):
    def test_package_matches_research_fit_and_preserves_demand(self):
        import subprocess, sys
        from export import assemble
        from price_train import fit_model
        from model import feature_frame
        from recipe import fit_price, weather_shape_names, abs_margin
        frame = synthetic(150)
        expected_x, expected_y = table(frame)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            package = folder/'pkg'
            assemble(dict(name='t', seeds=[9418], half_life_years=None, threshold=.57), package)
            frame.to_parquet(folder/'frame.parquet')
            run = subprocess.run([sys.executable, '-c', PACKAGE_SCRIPT, str(folder/'frame.parquet')],
                                 cwd=package, capture_output=True, text=True)
            self.assertIn('PACKAGE_OK', run.stdout, run.stderr[-3000:])
            x, y = pd.read_parquet(package/'x.parquet'), pd.read_parquet(package/'y.parquet')
            xs = pd.read_parquet(package/'xs.parquet')
            probs = json.loads((package/'probs.json').read_text())
        pd.testing.assert_frame_equal(x, expected_x, check_freq=False)
        pd.testing.assert_series_equal(y.price, expected_y.price, check_freq=False)
        # Harness construction: cached 571 table + feature_frame(weather_shape=True) columns, same order.
        shape = feature_frame(frame, weather_shape=True).loc[expected_x.index, weather_shape_names()].astype(np.float32)
        pd.testing.assert_frame_equal(xs, pd.concat([expected_x, shape], axis=1), check_freq=False)
        rows = x.loc[:x.index[-48*20 - 1]]
        self.assertEqual(len(rows) % 48, 0)
        for hl in ('None', '2.0'):
            config = dict(family='boosted', leaves=31, trees=5, learning_rate=.04, seed=9418)
            if hl != 'None':
                config['half_life_years'] = float(hl)
            bundle = fit_model('price', config, rows, expected_y.loc[rows.index])
            np.testing.assert_array_equal(bundle['estimator'].predict_proba(x.iloc[-48:])[:, 1], probs[hl])
        # Package weather-shape/params/weights members equal the research recipe on the harness inputs.
        self.assertEqual(probs['package_ws_columns'], list(xs.columns))
        params = dict(n_estimators=5, num_leaves=7, learning_rate=.05, colsample_bytree=.5, min_child_samples=40)
        margin = abs_margin(frame).loc[xs.index].to_numpy()
        research = [fit_price(xs, expected_y.price, s, params, None, .1, margin) for s in (9418, 9419)]
        np.testing.assert_array_equal(np.mean([m.predict_proba(xs.iloc[-48:])[:, 1] for m in research], axis=0),
                                      probs['package_ws'])


class RecipeTests(unittest.TestCase):
    def test_fit_price_equals_v2_fit_model(self):
        from price_train import fit_model
        from recipe import fit_price
        x, y = table(synthetic(150))
        for extra in ({}, {'half_life_years': 2.}):
            config = dict(family='boosted', leaves=31, trees=8, learning_rate=.04, seed=9417, **extra)
            reference = fit_model('price', config, x, y)['estimator'].predict_proba(x.iloc[-96:])[:, 1]
            ours = fit_price(x, y.price, 9417, dict(n_estimators=8), extra.get('half_life_years'))
            np.testing.assert_array_equal(ours.predict_proba(x.iloc[-96:])[:, 1], reference)

    def test_weather_shape_live_batch_parity_and_causality(self):
        from model import feature_frame, forecast_features
        from recipe import weather_shape_names
        frame = synthetic(70)
        names = weather_shape_names()
        self.assertEqual(len(names), 26)
        batch = feature_frame(frame, weather_shape=True)
        self.assertTrue(set(names) <= set(batch.columns))
        history, weather = frame.iloc[:-48], frame.iloc[-48:][WEATHER]
        live = forecast_features(history, weather, weather_shape=True)
        pd.testing.assert_frame_equal(batch.loc[live.index, names], live[names], check_freq=False)
        changed = frame.copy()
        changed.loc[changed.index[-48:], MARKET] = -1e9
        altered = feature_frame(changed, weather_shape=True)
        pd.testing.assert_frame_equal(batch.loc[live.index], altered.loc[live.index], check_freq=False)

    def test_abs_margin_definition(self):
        from recipe import abs_margin
        frame = synthetic(40)
        z = abs_margin(frame)
        days = market_days(frame.index)
        day = days.unique()[10]
        previous = frame.nsw_price[days == days.unique()[9]]
        tau = previous.mean()
        iqr = max(previous.quantile(.75) - previous.quantile(.25), 1)
        today = frame.nsw_price[days == day]
        np.testing.assert_allclose(z[days == day].to_numpy(), np.abs((today - tau) / iqr).to_numpy())
        self.assertTrue(z[days == days.unique()[0]].isna().all())


if __name__ == '__main__':
    unittest.main()
