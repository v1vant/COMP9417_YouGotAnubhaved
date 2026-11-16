"""Chronology, serialization and live forecasting checks for spline demand."""
import os
import sys
import tempfile
import unittest

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_forecasting import synthetic_data
from model import MARKET, WEATHER, market_days, training_table
from spline_ridge import SplineRidgeDemand


class SplineDemandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_data(days=80)
        cls.features, cls.truth = training_table(cls.frame)
        cls.days = market_days(cls.features.index)
        cls.cutoff = cls.days.unique()[-3]
        cls.train = cls.days < cls.cutoff
        cls.model = SplineRidgeDemand().fit(cls.features.loc[cls.train],
                                          cls.truth.loc[cls.train])

    def test_future_predictions_preserve_fitted_basis_and_scaling(self):
        features = self.features.loc[~self.train]
        knots = self.model.weather_.bsplines_[0].t.copy()
        means = self.model.linear_scaler_.mean_.copy()
        first = self.model.predict(features.iloc[:48])
        combined = self.model.predict(features)
        np.testing.assert_allclose(first, combined[:48])
        np.testing.assert_array_equal(knots, self.model.weather_.bsplines_[0].t)
        np.testing.assert_array_equal(means, self.model.linear_scaler_.mean_)
        self.assertTrue(np.isfinite(combined).all())
        with self.assertRaisesRegex(ValueError, "after training_end"):
            self.model.predict(self.features.loc[self.train].iloc[-48:])

    def test_serialized_live_and_batch_predictions_agree_without_future_markets(self):
        day = self.cutoff
        index = self.features.index[self.days == day]
        history = self.frame.loc[market_days(self.frame.index) < day]
        weather = self.frame.loc[index, WEATHER]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "demand.joblib")
            joblib.dump(self.model, path)
            restored = joblib.load(path)
            live = restored.predict_day(history, weather)
        np.testing.assert_allclose(live, self.model.predict(self.features.loc[index]), atol=1e-3)
        self.assertTrue(live.index.equals(index))
        changed = self.frame.copy()
        changed.loc[market_days(changed.index) >= day, MARKET] = -999999
        altered, _ = training_table(changed)
        np.testing.assert_allclose(self.model.predict(altered.loc[index]), live, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
