"""Solar features must be independent of batch extent and unavailable outcomes."""
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from test_forecasting import synthetic_data
from model import (Forecaster, MARKET, WEATHER, forecast_features, market_days,
                   predict_task, solar_trend_features, training_table)
from train import candidates, fit_model


class SolarTrendTests(unittest.TestCase):
    def test_signed_delta_and_fixed_date_origin_at_midnight(self):
        index = pd.DatetimeIndex(["2011-01-01 00:00", "2011-01-01 00:30"])
        x = pd.DataFrame({"syd_solar_change_1d": [100, -100],
                          "syd_solar_change_7d": [300, -300]}, index=index)
        f = solar_trend_features(x)
        np.testing.assert_array_equal(f.solar_delta_anchor, [200, -200])
        np.testing.assert_allclose(f.trend_years, [364 / 365.25, 365 / 365.25])
        np.testing.assert_allclose(f.solar_delta_trend,
                                   f.solar_delta_anchor * f.trend_years)

    def test_batch_live_and_future_observation_invariance(self):
        frame = synthetic_data()
        history, weather = frame.iloc[:-48], frame.iloc[-48:][WEATHER]
        x, _ = training_table(frame)
        expected = solar_trend_features(x).iloc[-48:]
        pd.testing.assert_frame_equal(expected,
                                      solar_trend_features(forecast_features(history, weather)))
        frame.loc[weather.index, MARKET] = -999999
        changed, _ = training_table(frame)
        pd.testing.assert_frame_equal(expected, solar_trend_features(changed).iloc[-48:])
        extended = synthetic_data(days=90)
        later = market_days(extended.index) > market_days(weather.index)[0]
        extended.loc[later, MARKET + WEATHER] = 999999
        extended_x, _ = training_table(extended)
        pd.testing.assert_frame_equal(expected,
                                      solar_trend_features(extended_x).loc[weather.index])

    def test_opt_in_save_load_and_price_unchanged(self):
        frame = synthetic_data()
        history, weather = frame.iloc[:-48], frame.iloc[-48:][WEATHER]
        x, y = training_table(history)
        config = dict(candidates("demand", quick=True, families=["boosted"])[0],
                      trees=3, holiday_context=True)
        base = fit_model("demand", config, x, y)
        demand = fit_model("demand", dict(config, solar_trend=True), x, y)
        price = fit_model("price", dict(config, solar_trend=True), x, y)
        self.assertEqual(len(demand["features"]), len(base["features"]) + 3)
        self.assertFalse(price["solar_trend"])
        self.assertNotIn("trend_years", price["features"])
        live = forecast_features(history, weather)
        expected = predict_task(demand, live)
        with tempfile.TemporaryDirectory() as directory:
            joblib.dump({"demand": demand, "price": price}, directory + "/models.joblib")
            output = Forecaster(directory).predict(history, weather)
        np.testing.assert_array_equal(output.nsw_demand_mw, expected)
        self.assertTrue(np.isfinite(output.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
