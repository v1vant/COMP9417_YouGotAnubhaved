"""Small regularised gate around a 50/50 demand blend."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler


def gate_features(features, lgbm, mlp):
    return pd.DataFrame({
        "hour_sin": features["hour_sin"],
        "hour_cos": features["hour_cos"],
        "prediction_gap": (lgbm-mlp) / np.maximum((lgbm+mlp)/2, 1),
        "temperature": features["syd_temp"],
        "solar": features["syd_solar"],
    }, index=features.index)


def fit_gate(context, lgbm, mlp, actual):
    actual = np.asarray(actual, dtype=float)
    lgbm, mlp = np.asarray(lgbm), np.asarray(mlp)
    if not (len(context) == len(lgbm) == len(mlp) == len(actual)):
        raise ValueError("Inputs must align")
    if not all(np.isfinite(v).all() for v in
               (context.to_numpy(), lgbm, mlp, actual)) or (actual <= 0).any():
        raise ValueError("Expected finite inputs and positive demand")
    scaler = StandardScaler()
    x = scaler.fit_transform(context)
    x = np.column_stack([np.ones(len(x)), x])

    def objective(theta):
        s = expit(x @ theta)
        weight = 0.25 + 0.5*s
        predicted = mlp + weight*(lgbm-mlp)
        error = 100*(predicted-actual)/actual
        # Smooth absolute percentage error, plus shrinkage toward 50/50.
        smooth = np.sqrt(error**2 + 0.01**2)
        penalty = 0.05
        loss = smooth.mean() + penalty*np.dot(theta, theta)
        derivative = error/smooth * 100*(lgbm-mlp)/actual * 0.5*s*(1-s)
        gradient = x.T @ derivative/len(x) + 2*penalty*theta
        return loss, gradient

    result = minimize(
        objective, np.zeros(x.shape[1]), jac=True,
        method="L-BFGS-B", options={"maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"Gate optimisation failed: {result.message}")
    return {
        "scaler": scaler, "coefficients": result.x,
        "columns": list(context.columns),
    }


def predict_gate(bundle, context, lgbm, mlp):
    x = bundle["scaler"].transform(context.loc[:, bundle["columns"]])
    x = np.column_stack([np.ones(len(x)), x])
    weight = 0.25 + 0.5*expit(x @ bundle["coefficients"])
    return weight*lgbm + (1-weight)*mlp, weight
