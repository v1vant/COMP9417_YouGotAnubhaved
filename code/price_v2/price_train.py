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
                   solar_trend_features, training_table, training_span, predict_price_probability)
import random_forest

SEED = 9417
DEFAULT_SELECTION = None
FAMILIES = ("linear", "boosted", "random_forest", "xgboost")
DEFAULT_FAMILIES = ("linear", "boosted")


def candidates(task, quick=False, families=DEFAULT_FAMILIES):
    linear = [{"family": "linear", "regularization": 10.0 if task == "demand" else 0.01}]
    boosted = [{"family": "boosted", "leaves": 31, "trees": 120 if quick else 600,
                "learning_rate": 0.04}]
    if not quick:
        linear.append({"family": "linear", "regularization": 1000.0 if task == "demand" else 0.1})
        boosted.append({"family": "boosted", "leaves": 63, "trees": 900,
                        "learning_rate": 0.025})
    unknown = set(families) - set(FAMILIES)
    if unknown or not families:
        raise ValueError(f"Expected model families from {FAMILIES}, got {families}")
    xgb = [{"family": "xgboost", "trees": 120 if quick else 600, "max_depth": 6,
            "learning_rate": .04, "min_child_weight": 50, "reg_lambda": 20}]
    configs = linear + boosted + random_forest.candidates(quick) + xgb
    return [config for config in configs if config["family"] in families]


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
                     LogisticRegression(C=config["regularization"], class_weight=config.get("class_weight", "balanced"),
                                        max_iter=1200, random_state=config.get("seed", SEED)))
        estimator = make_pipeline(StandardScaler(), estimator)
    elif config["family"] == "random_forest":
        columns = list(features.columns)
        estimator = random_forest.make_estimator(task, config, config.get("seed", SEED))
    elif config["family"] == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        columns = list(features.columns)
        cls = XGBRegressor if task == "demand" else XGBClassifier
        estimator = cls(n_estimators=config["trees"], max_depth=config.get("max_depth", 6),
                        learning_rate=config["learning_rate"], tree_method="hist", device="cpu",
                        min_child_weight=config.get("min_child_weight", 50),
                        reg_lambda=config.get("reg_lambda", 20), subsample=.8,
                        colsample_bytree=.8, n_jobs=2, random_state=config.get("seed", SEED),
                        objective="reg:absoluteerror" if task == "demand" else "binary:logistic")
    elif config["family"] == "boosted":
        columns = list(features.columns)
        cls = LGBMRegressor if task == "demand" else LGBMClassifier
        estimator = cls(
            objective="regression_l1" if task == "demand" else "binary",
            n_estimators=config["trees"], learning_rate=config["learning_rate"],
            num_leaves=config["leaves"], min_child_samples=config.get("min_child_samples", 160),
            max_depth=config.get("max_depth", -1), reg_alpha=config.get("reg_alpha", 0.),
            colsample_bytree=config.get("colsample_bytree", .85), subsample=config.get("subsample", .85), subsample_freq=1,
            reg_lambda=config.get("reg_lambda", 8.), max_bin=config.get("max_bin", 127), n_jobs=2, verbosity=-1,
            deterministic=True, force_col_wise=True, random_state=config.get("seed", SEED),
            **({"class_weight": config.get("class_weight", "balanced")} if task == "price" else {}),
        )
    else:
        raise ValueError(f"Unknown model family: {config['family']}")
    target = (truth["demand"].to_numpy() - features["demand_anchor"].to_numpy()
              if task == "demand" else truth["price"].to_numpy())
    kwargs = {}
    if task == "demand":
        weights = 1 / truth["demand"].to_numpy()
        weights /= weights.mean()
        kwargs["ridge__sample_weight" if config["family"] == "linear" else "sample_weight"] = weights
    if config.get("half_life_years") is not None and (
            not np.isfinite(config["half_life_years"]) or config["half_life_years"] <= 0):
        raise ValueError("half_life_years must be finite and positive")
    if task == "price" and (config.get("half_life_years") or config["family"] == "xgboost"):

        age = (market_days(features.index).max() - market_days(features.index)).days / 365.25
        weights = np.power(2., -np.asarray(age) / config.get("half_life_years", np.inf))
        if config.get("class_weight", "balanced") == "balanced":
            total_weight = weights.sum()
            for label in (0, 1):
                weights[target == label] *= total_weight / (2 * weights[target == label].sum())
            # Balance once, after recency weighting.
            if config["family"] != "xgboost":
                estimator.set_params(**({"logisticregression__class_weight": None}
                                     if config["family"] == "linear" else {"class_weight": None}))
        weights /= weights.mean()
        kwargs["logisticregression__sample_weight" if config["family"] == "linear" else "sample_weight"] = weights
    if task == "price" and config.get("early_stopping") and config["family"] == "boosted":
        from lightgbm import early_stopping
        split = market_days(features.index).max() - pd.Timedelta(days=89)
        older = market_days(features.index) < split
        if older.sum() == 0 or truth.loc[older, "price"].nunique() < 2:
            raise ValueError("Insufficient chronological early-stopping history")
        sample = kwargs.get("sample_weight")
        estimator.fit(features.loc[older, columns], target[older],
                      sample_weight=None if sample is None else sample[older],
                      eval_set=[(features.loc[~older, columns], target[~older])],
                      callbacks=[early_stopping(100, verbose=False)])
        estimator.set_params(n_estimators=max(1, estimator.best_iteration_))
    estimator.fit(features[columns], target, **kwargs)
    return {**span, "task": task, "config": config, "features": columns,
            "holiday_context": bool(config.get("holiday_context", False)),
            "solar_trend": use_solar_trend,
            "estimator": estimator, "threshold": config.get("threshold", 0.5)}


def select_models(features, truth, quick=False, families=DEFAULT_FAMILIES):
    days = market_days(features.index)
    train = days.year <= 2013
    valid = days.year == 2014
    if not train.any() or not valid.any():
        raise ValueError("Initial selection requires both 2010-2013 and 2014 data")
    selected, results, frozen, calibration = {}, [], {}, {}
    for task in ("demand", "price"):
        best_score = np.inf
        for number, config in enumerate(candidates(task, quick, families)):
            name = f"{task}_{config['family']}_{number}"
            model = fit_model(task, config, features.loc[train], truth.loc[train])
            if task == "price":
                probabilities = predict_price_probability(model, features.loc[valid])
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
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=DEFAULT_FAMILIES,
                        help="Candidate families used when --selection is not supplied")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "artifacts"),
                        help="Directory for models.joblib and manifest.json")
    args = parser.parse_args()
    started = time.monotonic()
    features, truth = training_table(load_data(args.csv))
    if args.selection:
        with open(args.selection, encoding="utf-8") as handle:
            selection = json.load(handle)
    elif DEFAULT_SELECTION is not None:
        selection = DEFAULT_SELECTION
    else:
        selected, scores, _, _ = select_models(features, truth, args.quick, args.families)
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
        if args.quick and config["family"] in ("boosted", "random_forest"):
            limit = 20 if config["family"] == "random_forest" else 120
            config["trees"] = min(config["trees"], limit)
        task_features = features
        if task == "price":
            from price_context import extend_price_features
            task_features = extend_price_features(load_data(args.csv), features,
                                config.get("price_feature_set", "base"))
        models[task] = fit_model(task, config, task_features.loc[mask], truth.loc[mask])
    output = args.output
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
