import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forecast_package"))
import model


class BlendTests(unittest.TestCase):
    def test_five_mlp_predictions_average_before_half_blend(self):
        forecaster = model.Forecaster.__new__(model.Forecaster)
        forecaster.models = {"demand": {}, "price": {}}
        forecaster.demand_mlp = {"seed": 9417}
        forecaster.demand_mlp_members = [{"seed": seed} for seed in range(9417, 9422)]
        weather = pd.DataFrame(index=pd.date_range("2026-01-01 00:30", periods=48, freq="30min"))
        labels = np.arange(48) % 2
        with patch.object(model, "forecast_features", return_value=pd.DataFrame()), \
             patch.object(model, "predict_task", side_effect=[np.full(48, 1000.), labels]), \
             patch.object(model, "predict_numpy_mlp", side_effect=[np.full(48, value) for value in [1000.,2000.,3000.,4000.,5000.]]):
            result = forecaster.predict(pd.DataFrame(), weather)
        np.testing.assert_array_equal(result.nsw_demand_mw, np.full(48, 2000.))
        np.testing.assert_array_equal(result.nsw_price_up, labels)

    def test_numpy_mlp_adds_residual_and_clips(self):
        features = pd.DataFrame({"demand_anchor": [7000., 19000.]})
        bundle = {"features": np.array(["demand_anchor"]), "mean": np.array([0.]),
                  "scale": np.array([1.])}
        for layer in (0, 2, 4):
            bundle[f"network_{layer}_weight"] = np.zeros((1,1), dtype=np.float32)
            bundle[f"network_{layer}_bias"] = np.zeros(1, dtype=np.float32)
        bundle["network_4_bias"][:] = 2
        np.testing.assert_array_equal(model.predict_numpy_mlp(bundle,features),[9000,20000])
        bundle["network_4_bias"][:] = -10
        np.testing.assert_array_equal(model.predict_numpy_mlp(bundle,features),[0,9000])

    def test_frozen_blend_and_price_passthrough(self):
        forecaster = model.Forecaster.__new__(model.Forecaster)
        forecaster.models = {"demand": {}, "price": {}}
        forecaster.demand_mlp = {"enabled": True}
        weather = pd.DataFrame(index=pd.date_range("2026-01-01 00:30", periods=48, freq="30min"))
        labels = np.arange(48)%2
        with patch.object(model,"forecast_features",return_value=pd.DataFrame()), \
             patch.object(model,"predict_task",side_effect=[np.full(48,1000.),labels]), \
             patch.object(model,"predict_numpy_mlp",return_value=np.full(48,3000.)):
            result = forecaster.predict(pd.DataFrame(),weather)
        np.testing.assert_array_equal(result.nsw_demand_mw,np.full(48,2000.))
        np.testing.assert_array_equal(result.nsw_price_up,labels)
        self.assertTrue(result.index.equals(weather.index))


if __name__ == "__main__":
    unittest.main()
