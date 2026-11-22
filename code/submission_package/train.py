"""Rebuild trained artifacts exclusively from the supplied CSV."""
import argparse
import json
import os
import platform
import time
import subprocess
import sys
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import load_data, market_days, predict_task, training_table

from c2_price import price_table, fit_c2

# Base seed for the single-model (non-ensemble) estimators.
SEED = 9417

# Frozen final configuration, used when no --selection file is supplied.
DEFAULT_SELECTION = {'demand': {'name': 'demand_single_lgbm_mlp5', 'config': {'family': 'boosted', 'leaves': 63, 'trees': 900, 'learning_rate': 0.025}, 'window_years': None, 'strategy': 'periodic_expanding', 'training_end': None, 'mlp_seeds': [9417, 9418, 9419, 9420, 9421]}, 'price': {'name': 'price_c2_bag5', 'config': {'name': 'C2_bag5_cutoff0.50', 'threshold': 0.5, 'half_life_years': None, 'seeds': [9417, 9418, 9419, 9420, 9421], 'weather_shape': True, 'params': {'num_leaves': 63, 'learning_rate': 0.02, 'n_estimators': 1200, 'colsample_bytree': 0.5, 'min_child_samples': 300}, 'source': 'code/price_v3/choose_candidate.py (round 2)', 'family': 'c2'}, 'window_years': None, 'strategy': 'periodic_expanding', 'training_end': None}, 'strategy_selection_end': '2024-12-31', 'quick': False}


def candidates(task, quick=False):
    """Return the original Stage 1 candidates; quick reduces the tree count."""
    # Baseline linear and boosted candidates.
    linear = [{"family": "linear", "regularization": 10.0 if task == "demand" else 0.01}]
    boosted = [{"family": "boosted", "leaves": 31, "trees": 120 if quick else 600,
                "learning_rate": 0.04}]

    # Extra candidates are skipped in quick mode.
    if not quick:
        linear.append({"family": "linear", "regularization": 1000.0 if task == "demand" else 0.1})
        boosted.append({"family": "boosted", "leaves": 63, "trees": 900,
                        "learning_rate": 0.025})
    return linear + boosted


def fit_model(task, config, features, truth):
    """Fit one task bundle with its feature schema and prediction threshold.

    Demand fits a weighted residual around the demand anchor. C2 price
    configurations delegate to the separate ensemble fitting routine.
    """
    # Keep the price-profile additions exclusive to classification.
    if task == "price" and config["family"] == "c2":
        return fit_c2(features, truth, config)
    if task == "demand":
        features = features.drop(columns=[
            col for col in features if col.startswith("nsw_price_") and (
                any(part in col for part in ("_profile_", "_before_margin_",
                                             "_ramp_", "_change_from_open"))
                or col == "nsw_price_weekly_margin")
        ])
    if not len(features):
        raise ValueError("No training observations")
    if task == "price" and truth["price"].nunique() < 2:
        raise ValueError("Classification training requires both classes")

    # Build the estimator for the chosen family.
    if config["family"] == "linear":
        # Linear models skip raw price levels and spread statistics.
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
        # LightGBM: L1 regression for demand, balanced binary for price.
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

    # Demand target is the residual around the anchor; price target is the label.
    target = (truth["demand"].to_numpy() - features["demand_anchor"].to_numpy()
              if task == "demand" else truth["price"].to_numpy())

    # Demand rows are weighted by inverse demand (percentage-error emphasis).
    kwargs = {}
    if task == "demand":
        weights = 1 / truth["demand"].to_numpy()
        weights /= weights.mean()
        kwargs["ridge__sample_weight" if config["family"] == "linear" else "sample_weight"] = weights
    estimator.fit(features[columns], target, **kwargs)
    return {"task": task, "config": config, "features": columns,
            "estimator": estimator, "threshold": config.get("threshold", 0.5)}


def select_models(features, truth, quick=False):
    """Fit through 2013, choose on 2014, then refit frozen candidate models."""
    # Train on 2010-2013 and validate on 2014.
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

            # Price: tune the probability cutoff for balanced accuracy.
            if task == "price":
                probabilities = model["estimator"].predict_proba(
                    features.loc[valid, model["features"]])[:, 1]
                thresholds = np.linspace(0.2, 0.8, 61)
                scores = [balanced_accuracy_score(truth.loc[valid, "price"],
                                                  probabilities > threshold)
                          for threshold in thresholds]
                model["threshold"] = float(thresholds[int(np.argmax(scores))])
                config = dict(config, threshold=model["threshold"])

            # Score on 2014: MAPE for demand, negative balanced accuracy for price (lower is better).
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

            # Refit on data through 2014 for the frozen version of this candidate.
            frozen[name] = fit_model(task, config, features.loc[days.year <= 2014],
                                     truth.loc[days.year <= 2014])

            # Keep the best candidate and its validation errors.
            if score < best_score:
                best_score = score
                selected[task] = {"name": name, "config": config}
                errors = (100 * np.abs(truth.loc[valid, "demand"].to_numpy() - predictions)
                          / truth.loc[valid, "demand"].to_numpy() if task == "demand" else
                          (truth.loc[valid, "price"].to_numpy() != predictions).astype(float) * 100)
                calibration[task] = errors
    return selected, results, frozen, calibration


def main():
    # Command-line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--selection", help="final_config.json produced by run_experiments.py")
    parser.add_argument("--quick", action="store_true", help="Smoke-test models, not final accuracy")
    args = parser.parse_args()
    started = time.monotonic()
    features, truth = training_table(load_data(args.csv))

    # Choose the model selection: supplied file, frozen default, or fresh search.
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
    # The selection controls each task's training end and optional sliding window.
    days = market_days(features.index)
    models = {}

    # C2 price uses its own feature table (with weather shape columns).
    price_features, price_truth = (price_table(load_data(args.csv))
        if selection["price"]["config"]["family"] == "c2" else (features, truth))

    # Fit the final model for each task.
    for task in ("demand", "price"):
        policy = selection[task]
        end = days.max() if not policy.get("training_end") else pd.Timestamp(policy["training_end"])
        mask = days <= end
        if policy.get("window_years"):
            mask &= days > end - pd.DateOffset(years=policy["window_years"])
        config = dict(policy["config"])
        if args.quick and config["family"] == "boosted":
            config["trees"] = min(config["trees"], 120)
        if task == "price" and config["family"] == "c2":
            if args.quick:
                config = dict(config, seeds=config['seeds'][:1], params=dict(config['params'], n_estimators=120))
            task_days = market_days(price_features.index)
            task_mask = task_days <= end
            if policy.get('window_years'):
                task_mask &= task_days > end - pd.DateOffset(years=policy['window_years'])
            models[task] = fit_model(task, config, price_features.loc[task_mask], price_truth.loc[task_mask])
        else:
            models[task] = fit_model(task, config, features.loc[mask], truth.loc[mask])
    # This is the package-local output directory, not the staged submission.
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(output, exist_ok=True)

    # Serialize locally, then replace atomically to avoid partial synced artifacts.
    with tempfile.TemporaryDirectory(prefix="forecast_artifact_") as directory:
        local = os.path.join(directory, "models.joblib")
        joblib.dump(models, local, compress=3)
        pending = os.path.join(output, "models.joblib.pending")
        shutil.copyfile(local, pending)
        os.replace(pending, os.path.join(output, "models.joblib"))

    # Record the run settings and library versions in a manifest.
    import lightgbm
    import sklearn
    manifest = {
        "seed": SEED, "data_end": str(days.max()), "selection": selection,
        "quick": args.quick, "seconds": time.monotonic() - started,
        "task_feature_counts": {task: len(bundle["features"])
                                for task, bundle in models.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__, "joblib": joblib.__version__},
    }
    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    # Train PyTorch in its own process; inference uses only exported NumPy weights.
    selection_path = os.path.join(output, "blend_training_selection.json")
    with open(selection_path, "w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "mlp_train.py"),
                    os.path.abspath(args.csv), "--output", output,
                    "--selection", selection_path], check=True)

    # Add the MLP blend details to the manifest and save it again.
    manifest["demand_blend"] = {"LightGBM": 0.5, "MLP": 0.5}
    manifest["mlp_epochs"] = 10
    with open(os.path.join(output, "demand_mlp_manifest.json"), encoding="utf-8") as handle:
        manifest["mlp_training"] = json.load(handle)
    manifest["demand_mlp_members"] = [member["file"] for member in manifest["mlp_training"]["members"]]
    manifest["demand_mlp_seeds"] = manifest["mlp_training"]["seeds"]
    manifest["inference"] = "LightGBM plus NumPy MLP; PyTorch used only in isolated training process"
    manifest["seconds"] = time.monotonic() - started
    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved artifacts to {output} in {manifest['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
