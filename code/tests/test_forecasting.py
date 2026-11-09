import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
import dayahead as da
from model import MARKET, WEATHER, feature_frame, forecast_features, market_days, training_table
from analysis_tools import OnlineDetector, confidence_intervals, metric_values
from run_experiments import baseline_predictions, simulate_policy


def synthetic_data(days=75, start="2014-11-01"):
    index = pd.date_range(pd.Timestamp(start) + pd.Timedelta(minutes=30),
                          periods=48 * days, freq="30min")
    slot = np.arange(len(index)) % 48
    day = np.arange(len(index)) // 48
    frame = pd.DataFrame(index=index)
    for number, col in enumerate(MARKET):
        frame[col] = (7000 + 400 * np.sin(slot / 48 * 2 * np.pi) + day * 3 + number
                      if col.endswith("demand") else 40 + 30 * np.cos(slot / 48 * 2 * np.pi) + day)
    for number, col in enumerate(WEATHER):
        frame[col] = 20 + number + 5 * np.sin(slot / 48 * 2 * np.pi)
    return frame


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_data()
        self.day = market_days(self.frame.index).unique()[-2]

    def test_batch_and_live_features_match(self):
        history, weather = da.make_inputs(self.frame, self.day)
        batch = feature_frame(self.frame).loc[weather.index]
        live = forecast_features(history, weather[WEATHER[::-1]])
        pd.testing.assert_frame_equal(batch, live)

    def test_future_market_and_later_weather_cannot_leak(self):
        original = feature_frame(self.frame)
        changed = self.frame.copy()
        today_and_future = market_days(changed.index) >= self.day
        changed.loc[today_and_future, MARKET] = -123456789
        changed.loc[market_days(changed.index) > self.day, WEATHER] = 987654321
        altered = feature_frame(changed)
        today = market_days(original.index) == self.day
        pd.testing.assert_frame_equal(original.loc[today], altered.loc[today])

    def test_targets_match_authoritative_harness(self):
        _, truth = training_table(self.frame)
        expected = da.targets(self.frame, self.day)
        np.testing.assert_array_equal(truth.loc[expected.index, "price"], expected["nsw_price_up"])
        np.testing.assert_array_equal(truth.loc[expected.index, "demand"], expected["nsw_demand_mw"])

    def test_baselines_use_previous_day_threshold(self):
        history, weather = da.make_inputs(self.frame, self.day)
        values = baseline_predictions(self.frame, weather.index)
        for lag in (1, 7):
            expected = da.baseline(history, weather, lag)
            np.testing.assert_array_equal(values[f"baseline_{lag}d"]["demand"].ravel(),
                                          expected["nsw_demand_mw"])
            np.testing.assert_array_equal(values[f"baseline_{lag}d"]["price"].ravel(),
                                          expected["nsw_price_up"])

    def test_midnight_belongs_to_previous_market_day(self):
        index = pd.DatetimeIndex(["2015-01-02 00:00", "2015-01-02 00:30"])
        self.assertEqual(list(market_days(index)), [pd.Timestamp("2015-01-01"),
                                                   pd.Timestamp("2015-01-02")])

    def test_rejects_nonconsecutive_forecast_day(self):
        history, weather = da.make_inputs(self.frame, self.day)
        weather.index += pd.Timedelta(minutes=30)
        with self.assertRaises(ValueError):
            forecast_features(history, weather)

    def test_strict_price_target_preserves_float64_precision(self):
        yesterday = market_days(self.frame.index) == self.day - pd.Timedelta(days=1)
        today = market_days(self.frame.index) == self.day
        self.frame.loc[yesterday, "nsw_price"] = 40.0000004
        self.frame.loc[today, "nsw_price"] = 40.0000002
        _, truth = training_table(self.frame)
        self.assertTrue((truth.loc[market_days(truth.index) == self.day, "price"] == 0).all())


class StatisticalTests(unittest.TestCase):
    def test_metrics_match_harness_in_single_class_period(self):
        values = metric_values([5000, 6000], [0, 0], [5100, 5900], [0, 1])
        self.assertEqual(values["balanced_accuracy"], da.balanced_accuracy([0, 0], [0, 1]))
        self.assertAlmostEqual(values["MAPE"], da.mape([5000, 6000], [5100, 5900]))

    def test_day_bootstrap_preserves_within_day_dependence(self):
        daily = np.arange(30, dtype=float)
        errors = np.repeat(daily[:, None], 48, axis=1)
        correct = np.repeat((daily < 15)[:, None], 48, axis=1)
        intervals = confidence_intervals(errors, correct)
        boot_width = intervals["MAPE_day_boot_high"] - intervals["MAPE_day_boot_low"]
        iid_width = intervals["MAPE_iid_high"] - intervals["MAPE_iid_low"]
        self.assertGreater(boot_width, iid_width * 3)

    def test_detector_uses_only_observed_days_and_cooldown(self):
        detector = OnlineDetector(np.zeros(365))
        day = pd.Timestamp("2015-01-01")
        self.assertTrue(detector.observe(day, 100))
        self.assertFalse(detector.observe(day + pd.Timedelta(days=1), 100))

    def test_drift_retraining_is_effective_the_following_day(self):
        frame = synthetic_data(days=75)
        features, truth = training_table(frame)
        days = market_days(features.index).unique()
        simulation_days = days[days >= pd.Timestamp("2015-01-01")]
        input_scores = pd.Series(100.0, index=simulation_days)
        fits = []

        def fitted(task, config, x, y):
            fits.append(market_days(x.index).max())
            return {}

        with patch("run_experiments.fit_model", side_effect=fitted), \
                patch("run_experiments.predict_task",
                      side_effect=lambda model, x: np.full(len(x), 7000.0)):
            _, retrains, _ = simulate_policy(
                "demand", "drift_sliding", {}, {}, features, truth, simulation_days,
                input_scores, np.zeros(365), np.zeros(365))
        self.assertEqual(retrains[0]["effective_day"], "2015-01-02")
        self.assertEqual(fits[0], pd.Timestamp("2015-01-01"))
        self.assertLessEqual(len(retrains), 6)


if __name__ == "__main__":
    unittest.main()
