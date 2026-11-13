"""Calendar context must agree between training batches and live inference."""
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from test_forecasting import synthetic_data
from model import (Forecaster, WEATHER, forecast_features, holiday_context_features,
                   market_days, training_table)
from train import candidates, fit_model


class HolidayContextTests(unittest.TestCase):
    def test_calendar_boundaries_use_market_days(self):
        index = pd.DatetimeIndex(["2024-12-20 00:30", "2025-01-01 00:00",
                                  "2025-01-02 00:30", "2025-01-05 00:30",
                                  "2025-01-06 00:30"])
        f = holiday_context_features(pd.DataFrame(index=index))
        self.assertEqual(f.holiday_tomorrow.iloc[1], 1)  # Dec 31 market day
        self.assertEqual(f.holiday_yesterday.iloc[2], 1)  # Jan 2
        self.assertEqual(f.christmas_shutdown.tolist(), [1, 1, 1, 1, 0])

    def test_batch_live_and_future_market_invariance(self):
        frame = synthetic_data(days=75, start="2024-10-20")
        history, weather = frame.iloc[:-48], frame.iloc[-48:][WEATHER]
        x, _ = training_table(frame)
        expected = holiday_context_features(x).iloc[-48:]
        pd.testing.assert_frame_equal(expected,
                                      holiday_context_features(forecast_features(history, weather)))
        frame.iloc[-48:, :] = -99999
        changed, _ = training_table(frame)
        cols = ["holiday_yesterday", "holiday_tomorrow", "christmas_shutdown"]
        pd.testing.assert_frame_equal(expected[cols], holiday_context_features(changed).iloc[-48:][cols])

    def test_opt_in_survives_serialization_and_live_prediction(self):
        frame = synthetic_data()
        history, weather = frame.iloc[:-48], frame.iloc[-48:][WEATHER]
        x, y = training_table(history)
        config = dict(candidates("demand", quick=True, families=["boosted"])[0], trees=3)
        models = {task: fit_model(task, dict(config, holiday_context=True), x, y)
                  for task in ("demand", "price")}
        self.assertIn("holiday_tomorrow", models["demand"]["features"])
        self.assertEqual(pd.Timestamp(models["demand"]["training_end"]), market_days(x.index).max())
        baseline = fit_model("demand", config, x, y)
        self.assertNotIn("holiday_tomorrow", baseline["features"])
        with tempfile.TemporaryDirectory() as directory:
            joblib.dump(models, directory + "/models.joblib")
            prediction = Forecaster(directory).predict(history, weather)
        self.assertEqual(prediction.shape, (48, 2))
        self.assertTrue(np.isfinite(prediction.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
