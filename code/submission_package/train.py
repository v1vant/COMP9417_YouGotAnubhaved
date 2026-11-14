"""Rebuild trained artifacts exclusively from the supplied CSV."""
import argparse
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import (holiday_context_features, load_data, market_days, predict_task,
                   solar_trend_features, training_table, training_span)

SEED = 9417
DEFAULT_SELECTION = {'demand': {'name': 'demand_boosted_3', 'config': {'family': 'boosted', 'leaves': 63, 'trees': 900, 'learning_rate': 0.025}, 'window_years': None, 'strategy': 'periodic_expanding', 'training_end': None}, 'price': {'name': 'price_boosted_2', 'config': {'family': 'boosted', 'leaves': 31, 'trees': 600, 'learning_rate': 0.04, 'threshold': 0.5700000000000001}, 'window_years': None, 'strategy': 'periodic_expanding', 'training_end': None}, 'strategy_selection_end': '2024-12-31', 'quick': False}


def candidates(task, quick=False):
    linear = [{"family": "linear", "regularization": 10.0 if task == "demand" else 0.01}]
    boosted = [{"family": "boosted", "leaves": 31, "trees": 120 if quick else 600,
                "learning_rate": 0.04}]
    if not quick:
        linear.append({"family": "linear", "regularization": 1000.0 if task == "demand" else 0.1})
        boosted.append({"family": "boosted", "leaves": 63, "trees": 900,
                        "learning_rate": 0.025})
    return linear + boosted


def fit_model(task, config, features, truth):
    if not len(features):
        raise ValueError("No training observations")
    span = training_span(features, truth)
    if config.get("holiday_context", False):
        features = holiday_context_features(features)
    use_solar_trend = task == "demand" and bool(config.get("solar_trend", False))
    if use_solar_trend:
        features = solar_trend_features(features)
    if task == "price" and truth["price"].nunique() < 2:
        raise ValueError("Classification training requires both classes")
    if config["family"] == "linear":
        columns = [col for col in features if not (
            ("price_lag_" in col and "_loglag_" not in col)
            or col.startswith("price_margin_")
            or ("price_" in col and any(word in col for word in ("min", "max", "std")))
        )]
        estimator = (Ridge(alpha=config["regularization"]) if task == "demand" else
                     LogisticRegression(C=config["regularization"], class_weight="balanced",
                                        max_iter=1200, random_state=SEED))
        estimator = make_pipeline(StandardScaler(), estimator)
    else:
        columns = list(features.columns)
        cls = LGBMRegressor if task == "demand" else LGBMClassifier
        estimator = cls(
            objective="regression_l1" if task == "demand" else "binary",
            n_estimators=config["trees"], learning_rate=config["learning_rate"],
            num_leaves=config["leaves"], min_child_samples=160,
            colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
            reg_lambda=8.0, max_bin=127, n_jobs=2, verbosity=-1,
            deterministic=True, force_col_wise=True, random_state=SEED,
            **({"class_weight": "balanced"} if task == "price" else {}),
        )
    target = (truth["demand"].to_numpy() - features["demand_anchor"].to_numpy()
              if task == "demand" else truth["price"].to_numpy())
    kwargs = {}
    if task == "demand":
        weights = 1 / truth["demand"].to_numpy()
        weights /= weights.mean()
        kwargs["ridge__sample_weight" if config["family"] == "linear" else "sample_weight"] = weights
    estimator.fit(features[columns], target, **kwargs)
    return {**span, "task": task, "config": config, "features": columns,
            "holiday_context": bool(config.get("holiday_context", False)),
            "solar_trend": use_solar_trend,
            "estimator": estimator, "threshold": config.get("threshold", 0.5)}


def select_models(features, truth, quick=False):
    days = market_days(features.index)
    train = days.year <= 2013
    valid = days.year == 2014
    if not train.any() or not valid.any():
        raise ValueError("Initial selection requires both 2010-2013 and 2014 data")
    selected, results, frozen, calibration = {}, [], {}, {}
    for task in ("demand", "price"):
        best_score = np.inf
        for number, config in enumerate(candidates(task, quick)):
            name = f"{task}_{config['family']}_{number}"
            model = fit_model(task, config, features.loc[train], truth.loc[train])
            if task == "price":
                probabilities = model["estimator"].predict_proba(
                    features.loc[valid, model["features"]])[:, 1]
                thresholds = np.linspace(0.2, 0.8, 61)
                scores = [balanced_accuracy_score(truth.loc[valid, "price"],
                                                  probabilities > threshold)
                          for threshold in thresholds]
                model["threshold"] = float(thresholds[int(np.argmax(scores))])
                config = dict(config, threshold=model["threshold"])
            predictions = predict_task(model, features.loc[valid])
            score = (np.mean(np.abs(truth.loc[valid, "demand"] - predictions)
                             / truth.loc[valid, "demand"]) * 100 if task == "demand" else
                     -balanced_accuracy_score(truth.loc[valid, "price"], predictions) * 100)
            metrics = (
                {"MAPE": float(score),
                 "MAE": float(np.mean(np.abs(truth.loc[valid, "demand"] - predictions)))}
                if task == "demand" else
                {"balanced_accuracy": float(-score),
                 "accuracy": float(np.mean(truth.loc[valid, "price"] == predictions) * 100)}
            )
            results.append({"model": name, "task": task, "config": config,
                            "validation_MAPE_or_negative_BA": float(score), **metrics})
            print(f"2014 validation {name}: {score:.3f}", flush=True)
            frozen[name] = fit_model(task, config, features.loc[days.year <= 2014],
                                     truth.loc[days.year <= 2014])
            if score < best_score:
                best_score = score
                selected[task] = {"name": name, "config": config}
                errors = (100 * np.abs(truth.loc[valid, "demand"].to_numpy() - predictions)
                          / truth.loc[valid, "demand"].to_numpy() if task == "demand" else
                          (truth.loc[valid, "price"].to_numpy() != predictions).astype(float) * 100)
                calibration[task] = errors
    return selected, results, frozen, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--selection", help="final_config.json produced by run_experiments.py")
    parser.add_argument("--quick", action="store_true", help="Smoke-test models, not final accuracy")
    args = parser.parse_args()
    started = time.monotonic()
    features, truth = training_table(load_data(args.csv))
    if args.selection:
        with open(args.selection, encoding="utf-8") as handle:
            selection = json.load(handle)
    elif DEFAULT_SELECTION is not None:
        selection = DEFAULT_SELECTION
    else:
        selected, scores, _, _ = select_models(features, truth, args.quick)
        selection = {
            task: dict(selected[task], window_years=4, training_end=None,
                       strategy="unvalidated_four_year_window")
            for task in selected
        }
        selection["validation"] = scores
        print("No simulation selection supplied: using an UNVALIDATED four-year final window.",
              flush=True)
    days = market_days(features.index)
    models = {}
    for task in ("demand", "price"):
        policy = selection[task]
        end = days.max() if not policy.get("training_end") else pd.Timestamp(policy["training_end"])
        mask = days <= end
        if policy.get("window_years"):
            mask &= days > end - pd.DateOffset(years=policy["window_years"])
        config = dict(policy["config"])
        if args.quick and config["family"] == "boosted":
            config["trees"] = min(config["trees"], 120)
        models[task] = fit_model(task, config, features.loc[mask], truth.loc[mask])
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(output, exist_ok=True)
    joblib.dump(models, os.path.join(output, "models.joblib"), compress=3)
    import lightgbm
    import sklearn
    manifest = {
        "seed": SEED, "data_end": str(days.max()), "selection": selection,
        "quick": args.quick, "seconds": time.monotonic() - started,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__, "joblib": joblib.__version__},
    }
    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved artifacts to {output} in {manifest['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
