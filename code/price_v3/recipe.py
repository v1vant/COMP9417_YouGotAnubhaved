"""Shared price recipe for the research harness and the submission package.

``fit_price`` reproduces code/price_v2/price_train.py:fit_model exactly for the
boosted price family (see test_v3.py), and adds three opt-in extensions that are
pre-registered in the round-2 plan:

* explicit LightGBM capacity parameters (``params``);
* weather-shape inputs: 26 Sydney temperature/solar shape columns that
  ``model.feature_frame(frame, weather_shape=True)`` already computes from the
  forecast day's supplied weather only (the caller appends them);
* near-threshold down-weighting, w = 1 - exp(-|z| / scale), where
  z = (P - tau) / max(previous-day IQR, 1). Training-time only: it uses realised
  prices of TRAINING days, never forecast-day observations.

This module has no imports from the research code so it can ship in a package.
"""
import numpy as np
import pandas as pd

BASE_PARAMS = dict(objective='binary', n_estimators=600, learning_rate=.04, num_leaves=31,
                   min_child_samples=160, max_depth=-1, reg_alpha=0., colsample_bytree=.85, subsample=.85,
                   subsample_freq=1, reg_lambda=8., max_bin=127, n_jobs=2, verbosity=-1, deterministic=True,
                   force_col_wise=True, class_weight='balanced')
TUNABLE = {'n_estimators', 'learning_rate', 'num_leaves', 'min_child_samples', 'colsample_bytree', 'reg_lambda'}


def weather_shape_names():
    """The 26 columns added by model.feature_frame(..., weather_shape=True), in a fixed order."""
    names = []
    for col in ('syd_temp', 'syd_solar'):
        for offset in (2, 4):
            names += [f'{col}_before_{offset}', f'{col}_after_{offset}', f'{col}_ramp_{offset}',
                      f'{col}_ramp_{offset}_change_1d']
        names += [f'{col}_daily_{stat}_change_1d' for stat in ('mean', 'min', 'max')]
        names += [f'{col}_peak_slot', f'{col}_peak_slot_change_1d']
    return names


def market_days(index):
    return (index - pd.Timedelta(minutes=30)).normalize()


def abs_margin(frame):
    """|P - tau| / max(IQR of day D, 1) for every interval of day D+1 (harness-exact tau)."""
    days = market_days(frame.index)
    price = frame['nsw_price']
    means = price.groupby(days).agg(lambda s: s.mean())
    quartiles = price.groupby(days).quantile([.25, .75]).unstack()
    scale = (quartiles[.75] - quartiles[.25]).clip(lower=1)
    lookup = pd.Series(days, index=frame.index)
    tau, iqr = lookup.map(means.shift(1)), lookup.map(scale.shift(1))
    return ((price - tau) / iqr).abs()


def fit_price(x, labels, seed, params=None, half_life_years=None, near_threshold_scale=None, margin=None):
    """One LightGBM price member. Weights are balanced once, after all multiplicative weighting."""
    from lightgbm import LGBMClassifier
    params = dict(params or {})
    unknown = set(params) - TUNABLE
    if unknown:
        raise ValueError(f'Unsupported LightGBM parameters: {sorted(unknown)}')
    if half_life_years is not None and (not np.isfinite(half_life_years) or half_life_years <= 0):
        raise ValueError('half_life_years must be finite and positive')
    settings = dict(BASE_PARAMS, **params, random_state=seed)
    target = np.asarray(labels).astype(int)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError('Classification training requires both classes')
    weights = None
    if half_life_years:
        days = market_days(x.index)
        age = np.asarray((days.max() - days).days) / 365.25
        weights = np.power(2., -age / half_life_years)
    if near_threshold_scale:
        if margin is None or len(margin) != len(x):
            raise ValueError('near_threshold_scale requires one |z| value per training row')
        margin = np.asarray(margin, dtype=float)
        if not np.isfinite(margin).all():
            raise ValueError('Nonfinite threshold margins')
        near = 1 - np.exp(-margin / near_threshold_scale)
        weights = near if weights is None else weights * near
    kwargs = {}
    if weights is not None:
        total = weights.sum()
        for label in (0, 1):
            weights[target == label] *= total / (2 * weights[target == label].sum())
        settings['class_weight'] = None
        kwargs['sample_weight'] = weights / weights.mean()
    return LGBMClassifier(**settings).fit(x, target, **kwargs)
