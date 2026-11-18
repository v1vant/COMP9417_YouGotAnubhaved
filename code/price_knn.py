"""Standalone distance-weighted KNN price classifier."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

PRICE_TIME = [
    "hour_sin", "hour_cos", "annual_sin", "annual_cos",
    "weekend", "holiday",
    "price_margin_1", "price_margin_2", "price_margin_7",
    "price_threshold", "nsw_price_previous_std",
]
FULL = PRICE_TIME + [
    "nsw_demand_lag_1", "nsw_demand_lag_7", "nsw_demand_recent_change",
    "syd_temp", "syd_solar", "syd_cloud", "syd_rh",
]


def knn_features(features, feature_set="full"):
    columns = {"full": FULL, "price_time": PRICE_TIME}[feature_set]
    x = features.loc[:, columns].astype(float).copy()
    for column in (
        "price_margin_1", "price_margin_2", "price_margin_7",
        "price_threshold",
    ):
        x[column] = np.sign(x[column]) * np.log1p(np.abs(x[column]))
    if x["nsw_price_previous_std"].lt(0).any():
        raise ValueError("Price standard deviation cannot be negative")
    x["nsw_price_previous_std"] = np.log1p(x["nsw_price_previous_std"])
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("KNN features must be finite")
    return x


def fit_price_knn(features, labels, neighbors=75, feature_set="full"):
    if features.empty or not features.index.equals(labels.index):
        raise ValueError("Expected nonempty aligned training inputs")
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Training requires both binary classes")
    if not 1 <= neighbors <= len(features):
        raise ValueError("Invalid neighbour count")
    model = make_pipeline(
        FunctionTransformer(knn_features, kw_args={"feature_set": feature_set}),
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=neighbors, weights="distance",
            algorithm="brute", metric="euclidean", n_jobs=2,
        ),
    )
    model.fit(features, labels)
    return model


def predict_price_knn(model, features):
    classes = model.classes_.tolist()
    return model.predict_proba(features)[:, classes.index(1)]
