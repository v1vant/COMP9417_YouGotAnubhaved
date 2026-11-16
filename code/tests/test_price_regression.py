import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from price_regression import fit_price_regressor, predict_price_regressor


class ConstantEstimator:
    def predict(self, features):
        return np.array([-20.0, 40.0, 41.0])


class PriceRegressionTests(unittest.TestCase):
    def test_negative_prices_and_strict_threshold(self):
        features = pd.DataFrame({"x": [0, 1, 2]})
        bundle = {"estimator": ConstantEstimator(), "features": ["x"]}
        result = predict_price_regressor(bundle, features, pd.Series([-10., 40., 40.]))
        np.testing.assert_array_equal(result.nsw_price_pred, [-20, 40, 41])
        np.testing.assert_array_equal(result.nsw_price_margin, [-10, 0, 1])
        np.testing.assert_array_equal(result.nsw_price_up, [0, 0, 1])

    def test_actual_price_scale_is_fitted(self):
        features = pd.DataFrame({"x": np.arange(400, dtype=float)})
        prices = pd.Series(np.full(400, -25.0))
        model = fit_price_regressor(features, prices,
                                    {"trees": 2, "leaves": 3, "learning_rate": 0.04})
        result = predict_price_regressor(model, features, pd.Series(np.zeros(400)))
        np.testing.assert_allclose(result.nsw_price_pred, -25)

    def test_rejects_misaligned_targets(self):
        with self.assertRaises(ValueError):
            fit_price_regressor(pd.DataFrame({"x": [1]}, index=[0]),
                                pd.Series([20.], index=[1]), {})


if __name__ == "__main__":
    unittest.main()
