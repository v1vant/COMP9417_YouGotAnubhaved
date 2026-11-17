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


class NaiveBayesTests(unittest.TestCase):
    def test_naive_bayes_fits_and_predicts_binary_labels(self):
        from model import predict_task
        from train import candidates, fit_model
        features, truth = training_table(synthetic_data(days=75))
        [config] = [c for c in candidates("price") if c["family"] == "naive_bayes"]
        model = fit_model("price", config, features, truth)
        predictions = predict_task(model, features)
        self.assertEqual(len(predictions), len(features))
        self.assertTrue(set(np.unique(predictions)) <= {0, 1})
        # Synthetic prices follow a fixed daily shape, so slot-aware NB should beat chance.
        self.assertGreater(np.mean(predictions == truth["price"].to_numpy()), 0.6)

    def test_naive_bayes_matches_bayes_rule_by_hand(self):
        from scipy.stats import norm
        from model import naive_bayes_probability
        from model import fit_naive_bayes
        features, truth = training_table(synthetic_data(days=75))
        config = {"discrete": ["slot"], "continuous": ["demand_anchor"], "alpha": 1.0}
        params = fit_naive_bayes(config, features, truth["price"])
        row, y = features.iloc[[100]], truth["price"].to_numpy()
        slot, x = int(row["slot"].iloc[0]), row["demand_anchor"].iloc[0]
        joint = []
        for k in (0, 1):
            in_class = features[y == k]
            prior = np.mean(y == k)
            p_slot = (np.sum(in_class["slot"] == slot) + 1) / (len(in_class) + 48)  # Laplace
            values = in_class["demand_anchor"].to_numpy(np.float64)
            p_x = norm.pdf(x, values.mean(), np.sqrt(values.var() + 1e-9 * params["var"].max()))
            joint.append(prior * p_slot * p_x)
        self.assertAlmostEqual(naive_bayes_probability(params, row)[0],
                               joint[1] / sum(joint), places=6)

    def test_zero_inflated_feature_uses_hurdle_likelihood(self):
        from scipy.stats import norm
        from model import naive_bayes_terms
        from model import fit_naive_bayes
        features, truth = training_table(synthetic_data(days=75))
        params = fit_naive_bayes({"discrete": [], "continuous": ["syd_cool"], "alpha": 1.0},
                                 features, truth["price"])
        self.assertTrue(params["zero_inflated"][0])            # syd_cool is often exactly 0
        cool, y = features["syd_cool"].to_numpy(np.float64), truth["price"].to_numpy()
        terms = naive_bayes_terms(params, features)[:, 0, :]
        for row in (int(np.flatnonzero(cool == 0)[0]), int(np.flatnonzero(cool > 0)[0])):
            for k in (0, 1):
                values = cool[y == k]
                p_zero = (np.sum(values == 0) + 1) / (len(values) + 2)
                nonzero = values[values != 0]
                expected = (np.log(p_zero) if cool[row] == 0 else
                            np.log(1 - p_zero) + norm.logpdf(
                                cool[row], nonzero.mean(),
                                np.sqrt(nonzero.var() + 1e-9 * params["var"].max())))
                self.assertAlmostEqual(terms[row, k], expected, places=6)

    def test_naive_bayes_is_not_a_demand_candidate(self):
        from train import candidates
        self.assertFalse(any(c["family"] == "naive_bayes" for c in candidates("demand")))


class SVMTests(unittest.TestCase):
    def setUp(self):
        self.features, self.truth = training_table(synthetic_data(days=75))

    def test_svm_predicts_binary_labels_with_all_or_listed_features(self):
        from model import predict_task
        from train import fit_model
        base = {"family": "svm", "kernel": "rbf", "gamma": 1.0, "C": 0.3, "max_rows": 800}
        for config in (base, dict(base, features=["slot", "price_margin_1", "nsw_demand_lag_1"])):
            model = fit_model("price", config, self.features, self.truth)
            if config.get("features"):
                self.assertEqual(model["features"], config["features"])
            predictions = predict_task(model, self.features)
            self.assertEqual(len(predictions), len(self.features))
            self.assertTrue(set(np.unique(predictions)) <= {0, 1})
            self.assertGreater(np.mean(predictions == self.truth["price"].to_numpy()), 0.6)

    def test_exact_svm_subsample_is_capped_and_reproducible(self):
        from train import fit_model
        config = {"family": "svm", "kernel": "rbf", "gamma": 1.0, "C": 0.3, "max_rows": 500}
        first = fit_model("price", config, self.features, self.truth)["estimator"][-1]
        second = fit_model("price", config, self.features, self.truth)["estimator"][-1]
        self.assertLessEqual(first.n_support_.sum(), 500)
        np.testing.assert_array_equal(first.support_, second.support_)

    def test_svm_is_not_a_demand_candidate(self):
        from train import candidates
        self.assertFalse(any(c["family"] == "svm" for c in candidates("demand")))
        self.assertTrue(any(c["family"] == "svm" for c in candidates("price")))


class NeuralNetTests(unittest.TestCase):
    CONFIG = {"family": "neural_net", "hidden": [16, 8], "dropout": 0.5,
              "learning_rate": 3e-3, "weight_decay": 1e-4, "batch_size": 128,
              "epochs": 20, "seeds": [0, 1]}

    @classmethod
    def setUpClass(cls):
        from train import fit_model
        cls.features, cls.truth = training_table(synthetic_data(days=75))
        cls.model = fit_model("price", cls.CONFIG, cls.features, cls.truth)

    def test_predicts_binary_labels_better_than_chance(self):
        from model import predict_task
        predictions = predict_task(self.model, self.features)
        self.assertEqual(len(predictions), len(self.features))
        self.assertTrue(set(np.unique(predictions)) <= {0, 1})
        self.assertGreater(np.mean(predictions == self.truth["price"].to_numpy()), 0.6)

    def test_backpropagation_matches_finite_differences(self):
        from train import neural_net_loss_gradients
        rng = np.random.default_rng(0)
        layers = [[rng.normal(size=(6, 5)), rng.normal(size=6)],
                  [rng.normal(size=(4, 6)), rng.normal(size=4)],
                  [rng.normal(size=(1, 4)), rng.normal(size=1)]]
        x, y = rng.normal(size=(12, 5)), (rng.random(12) < 0.4).astype(float)
        masks = [(rng.random((12, 6)) < 0.5) / 0.5, (rng.random((12, 4)) < 0.5) / 0.5]
        for mask in (None, masks):
            _, gradients = neural_net_loss_gradients(layers, x, y, 1.7, mask)
            for layer, grads in zip(layers, gradients):
                for parameter, gradient in zip(layer, grads):
                    numeric = np.zeros_like(parameter)
                    for index in np.ndindex(parameter.shape):
                        original = parameter[index]
                        parameter[index] = original + 1e-6
                        up = neural_net_loss_gradients(layers, x, y, 1.7, mask)[0]
                        parameter[index] = original - 1e-6
                        down = neural_net_loss_gradients(layers, x, y, 1.7, mask)[0]
                        parameter[index] = original
                        numeric[index] = (up - down) / 2e-6
                    np.testing.assert_allclose(gradient, numeric, atol=1e-6)

    def test_prediction_matches_formula_by_hand(self):
        from model import neural_net_probability, svm_inputs
        params = self.model["estimator"]
        row = self.features.iloc[[100]]
        z = (svm_inputs(params["columns"], row)[0] - params["mean"]) / params["std"]
        expected = []
        for (w1, b1), (w2, b2), (w3, b3) in params["members"]:
            h1 = np.maximum(w1 @ z + b1, 0)
            h2 = np.maximum(w2 @ h1 + b2, 0)
            expected.append(1 / (1 + np.exp(-(w3 @ h2 + b3)[0])))
        self.assertAlmostEqual(neural_net_probability(params, row)[0], np.mean(expected),
                               places=10)

    def test_training_is_deterministic_and_seeds_differ(self):
        from train import fit_model
        again = fit_model("price", self.CONFIG, self.features, self.truth)["estimator"]
        first = self.model["estimator"]
        for member, repeat in zip(first["members"], again["members"]):
            for (weight, bias), (weight2, bias2) in zip(member, repeat):
                np.testing.assert_array_equal(weight, weight2)
                np.testing.assert_array_equal(bias, bias2)
        self.assertFalse(np.array_equal(first["members"][0][0][0], first["members"][1][0][0]))

    def test_artifact_is_plain_numpy_and_survives_joblib(self):
        import io
        import joblib
        from model import neural_net_probability
        params = self.model["estimator"]
        self.assertEqual([w.shape for w, _ in params["members"][0]],
                         [(16, len(params["columns"])), (8, 16), (1, 8)])
        for member in params["members"]:
            for weight, bias in member:
                self.assertIs(type(weight), np.ndarray)
                self.assertIs(type(bias), np.ndarray)
        buffer = io.BytesIO()
        joblib.dump(self.model, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_array_equal(neural_net_probability(restored["estimator"], self.features),
                                      neural_net_probability(params, self.features))

    def test_standardisation_uses_training_rows_only(self):
        from model import neural_net_inputs
        z = neural_net_inputs(self.model["estimator"], self.features)
        varying = z.std(axis=0) > 1e-6   # constant columns (up to rounding) stay unscaled
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(z.std(axis=0)[varying], 1, atol=1e-6)

    def test_neural_net_is_a_price_only_candidate(self):
        from train import candidates
        self.assertFalse(any(c["family"] == "neural_net" for c in candidates("demand")))
        self.assertTrue(any(c["family"] == "neural_net" for c in candidates("price")))


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
