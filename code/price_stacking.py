"""Small price stacker fitted on chronological out-of-sample predictions."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def stacking_features(classifier_probability, predicted_margin):
    if not classifier_probability.index.equals(predicted_margin.index):
        raise ValueError("Stacking inputs must have matching indices")
    features = pd.DataFrame({
        "classifier_probability": classifier_probability,
        "predicted_margin": predicted_margin,
    })
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Stacking inputs must be finite")
    if not classifier_probability.between(0, 1).all():
        raise ValueError("Classifier probabilities must be between 0 and 1")
    return features


def fit_stacker(oof_probability, oof_margin, labels):
    # Each input prediction must come from base models trained strictly
    # before its forecast day. The chronological runner enforces this.
    features = stacking_features(oof_probability, oof_margin)
    if not features.index.equals(labels.index):
        raise ValueError("Labels must align with stacking inputs")
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Training requires both binary classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=9417,
        ),
    )
    model.fit(features, labels)
    return model


def predict_stacker(model, classifier_probability, predicted_margin):
    features = stacking_features(classifier_probability, predicted_margin)
    probability = model.predict_proba(features)[:, 1]
    return pd.DataFrame({
        "stacked_probability": probability,
        "nsw_price_up": (probability > 0.5).astype(np.int64),
    }, index=features.index)
