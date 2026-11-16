"""Separate actual-price regression candidate; does not alter Forecaster."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def fit_price_regressor(features, prices, config):
    if features.empty or not features.index.equals(prices.index):
        raise ValueError("Expected nonempty features aligned with actual prices")
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Actual prices must be finite")
    estimator = LGBMRegressor(
        objective="regression_l1", n_estimators=config["trees"],
        learning_rate=config["learning_rate"], num_leaves=config["leaves"],
        min_child_samples=160, colsample_bytree=0.85, subsample=0.85,
        subsample_freq=1, reg_lambda=8.0, max_bin=127, n_jobs=2,
        verbosity=-1, deterministic=True, force_col_wise=True, random_state=9417,
    )
    # Fit actual $/MWh values, including negatives, rather than class labels.
    estimator.fit(features, prices)
    return {"estimator": estimator, "features": list(features.columns),
            "config": dict(config), "target": "actual_nsw_price", "loss": "L1"}


def predict_price_regressor(bundle, features, previous_day_mean):
    if not features.index.equals(previous_day_mean.index):
        raise ValueError("Previous-day means must align with prediction rows")
    price = bundle["estimator"].predict(features.loc[:, bundle["features"]])
    threshold = previous_day_mean.to_numpy(dtype=float)
    if not np.isfinite(price).all() or not np.isfinite(threshold).all():
        raise ValueError("Predictions and previous-day means must be finite")
    return pd.DataFrame({
        "nsw_price_pred": price,
        "nsw_price_margin": price - threshold,
        "nsw_price_up": (price > threshold).astype(np.int64),
    }, index=features.index)
