"""Submission inference.

Demand: the unchanged reference 50/50 LightGBM / NumPy-MLP blend, computed from
exactly the same feature call as base_model.Forecaster. Price: mean probability
of the frozen LightGBM members (optionally with the 26 weather-shape inputs,
which use only the supplied next-day weather), compared with a fixed cutoff.
No fitted parameter changes here.
"""
import numpy as np
import pandas as pd

from base_model import *  # noqa: F401,F403  (reference_train.py and mlp_train.py import from model)
from base_model import Forecaster as ReferenceForecaster
from base_model import forecast_features, predict_numpy_mlp, predict_task


def price_probability(bundle, features):
    x = features.loc[:, bundle['features']]
    members = bundle.get('members') or [bundle['estimator']]
    return np.mean([member.predict_proba(x)[:, 1] for member in members], axis=0)


def price_features(bundle, history, weather_next, demand_features):
    if bundle.get('config', {}).get('weather_shape'):
        return forecast_features(history, weather_next, weather_shape=True)
    return demand_features


class Forecaster(ReferenceForecaster):
    def __init__(self, artifact_dir: str):
        super().__init__(artifact_dir)
        for member in self.models['price'].get('members', []):
            if 'n_jobs' in member.get_params():
                member.set_params(n_jobs=2)

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        features = forecast_features(history, weather_next)
        bundle = self.models['price']   # no date guard: check_submission scores days inside the training range
        demand = predict_task(self.models['demand'], features)
        if self.demand_mlp is not None:
            demand = 0.5*demand + 0.5*predict_numpy_mlp(self.demand_mlp, features)
        probability = price_probability(bundle, price_features(bundle, history, weather_next, features))
        return pd.DataFrame({
            'nsw_demand_mw': demand,
            'nsw_price_up': (probability > bundle['threshold']).astype(np.int64),
        }, index=weather_next.index)
