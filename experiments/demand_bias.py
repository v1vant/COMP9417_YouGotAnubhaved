"""Prototype, stateless demand correction based on recent out-of-sample errors.

Pass the demand model's training end market day. When fewer than 28 full days
have elapsed after training, fall back to the uncorrected prediction. This is
research code and has not been integrated into a submission package.
"""
import numpy as np
import pandas as pd


def corrected_demand(model_api, forecaster, history, weather_next, training_end):
    m = model_api
    if forecaster.demand_mlp is None:
        raise ValueError('Expected the 50/50 demand blend')
    features = m.forecast_features(history, weather_next)
    demand = 0.5*m.predict_task(forecaster.models['demand'], features)
    demand += 0.5*m.predict_numpy_mlp(forecaster.demand_mlp, features)
    # The interval at history[-28*48] is the first end-stamp of the
    # earliest correction day. Subtract 30 minutes to recover its market day.
    if len(history) < 63*48:
        return demand
    first_day = (history.index[-28*48] - pd.Timedelta(minutes=30)).normalize()
    if first_day <= pd.Timestamp(training_end).normalize():
        return demand
    past = history.iloc[-63*48:][m.MARKET + m.WEATHER]
    recent = m.feature_frame(past).loc[history.index[-28*48:]]
    predicted = 0.5*m.predict_task(forecaster.models['demand'], recent)
    predicted += 0.5*m.predict_numpy_mlp(forecaster.demand_mlp, recent)
    errors = history.nsw_demand.iloc[-28*48:].to_numpy() - predicted
    return np.clip(demand + 0.5*errors.reshape(28, 48).mean(axis=0), 0, 20000)
