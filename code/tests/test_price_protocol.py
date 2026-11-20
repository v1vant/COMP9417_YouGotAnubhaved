"""The C2 protocol path must respect the Stage 1 split and never use later data."""
import tempfile
import unittest

import numpy as np
import pandas as pd

from test_forecasting import synthetic_data
from model import MARKET, WEATHER, market_days, training_table
import run_price_protocol as protocol

TINY = {"name": "price_c2_test", "seeds": [1, 2], "params": {"n_estimators": 5}}


def tables(frame):
    features, truth = training_table(frame)
    return protocol.c2_table(frame, features.index), truth


class PriceProtocolTests(unittest.TestCase):
    def test_c2_stage1_fits_through_2013_then_2014(self):
        x, truth = tables(synthetic_data(days=760, start="2013-01-01"))
        ends = []

        def recording(task, config, xs, ys):
            ends.append(market_days(xs.index).max().year)
            return protocol.fit_bag(task, config, xs, ys)

        result = protocol.stage1(x, truth, TINY, fit=recording)
        self.assertEqual(ends, [2013, 2014])
        self.assertIn(round(result["cutoff"], 2), np.round(protocol.CUTOFF_GRID, 2))
        flipped = truth.copy()
        later = market_days(flipped.index).year >= 2015
        self.assertTrue(later.any())
        flipped.loc[later, "price"] = 1 - flipped.loc[later, "price"]
        again = protocol.stage1(x, flipped, TINY)
        self.assertEqual(result["cutoff"], again["cutoff"])
        self.assertEqual(result["scores"], again["scores"])

    def test_c2_policy_ignores_future_data(self):
        frame = synthetic_data(days=760, start="2013-01-01")
        changed = frame.copy()
        cutoff = pd.Timestamp("2015-01-15")
        changed.loc[market_days(changed.index) >= cutoff, MARKET + WEATHER] *= 10
        runs = []
        for data in (frame, changed):
            x, truth = tables(data)
            stage = protocol.stage1(x, truth, TINY)
            feature_days = market_days(x.index)
            days = feature_days[feature_days >= pd.Timestamp("2015-01-01")].unique()
            for mode in ("tuned_2014", "fixed_0.50"):
                result, retrains, signals = protocol.run_policy(
                    "drift_sliding", mode, stage, TINY, x, truth, days,
                    pd.Series(100.0, index=days), np.zeros(365))
                self.assertTrue(retrains)
                self.assertLessEqual(len(retrains), 6)
                for retrain in retrains:
                    self.assertLess(pd.Timestamp(retrain["training_end"]),
                                    pd.Timestamp(retrain["effective_day"]))
                before = days < cutoff
                runs.append((result[before], retrains,
                             [s for s in signals if pd.Timestamp(s["day"]) < cutoff]))
        for original, perturbed in zip(runs[:2], runs[2:]):
            np.testing.assert_array_equal(original[0], perturbed[0])
            self.assertEqual(original[1], perturbed[1])
            self.assertEqual(original[2], perturbed[2])

    def test_worker_job_reads_saved_tables_and_matches_direct_run(self):
        x, truth = tables(synthetic_data(days=760, start="2013-01-01"))
        stage = protocol.stage1(x, truth, TINY)
        feature_days = market_days(x.index)
        days = feature_days[feature_days >= pd.Timestamp("2015-01-01")].unique()
        scores = pd.Series(100.0, index=days)
        direct = protocol.run_policy("drift_sliding", "tuned_2014", stage, TINY, x, truth, days,
                                     scores, np.zeros(365))
        with tempfile.TemporaryDirectory() as cache:
            protocol.save_tables(cache, x, truth)
            worker = protocol._policy_job(("drift_sliding", "tuned_2014", stage, TINY, cache,
                                           scores, np.zeros(365)))
        np.testing.assert_array_equal(direct[0], worker[0])
        self.assertEqual(direct[1], worker[1])
        self.assertEqual(direct[2], worker[2])


if __name__ == "__main__":
    unittest.main()
