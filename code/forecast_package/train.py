"""Rebuild trained artifacts exclusively from the supplied CSV."""
import argparse
import json
import os
import platform
import time
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model import (NAIVE_BAYES, NEURAL_NET, SVM, fit_naive_bayes, load_data, market_days,
                   neural_net_columns, neural_net_log_mask, neural_net_transform,
                   predict_task, price_probability, svm_columns, svm_inputs, training_table)

SEED = 9417
DEFAULT_SELECTION = {'demand': {'name': 'demand_boosted_3', 'config': {'family': 'boosted', 'leaves': 63, 'trees': 900, 'learning_rate': 0.025}, 'window_years': None, 'strategy': 'periodic_expanding', 'training_end': None}, 'price': {'name': 'price_boosted_2', 'config': {'family': 'boosted', 'leaves': 31, 'trees': 600, 'learning_rate': 0.04, 'threshold': 0.5700000000000001}, 'window_years': None, 'strategy': 'periodic_expanding', 'training_end': None}, 'strategy_selection_end': '2024-12-31', 'quick': False}


def candidates(task, quick=False):
    linear = [{"family": "linear", "regularization": 10.0 if task == "demand" else 0.01}]
    boosted = [{"family": "boosted", "leaves": 31, "trees": 120 if quick else 600,
                "learning_rate": 0.04}]
    # Naive Bayes and the SVM are classifiers, so they are only candidates for price direction.
    bayes = [dict(NAIVE_BAYES)] if task == "price" else []
    # RBF kernel chosen on 2014 over linear, polynomial and sigmoid kernels; gamma is in
    # units of 1/#features.
    svm = ([dict(SVM, max_rows=2000 if quick else SVM["max_rows"])]
           if task == "price" else [])
    neural = ([dict(NEURAL_NET, **({"seeds": [0], "epochs": 3} if quick else {}))]
              if task == "price" else [])
    if not quick:
        linear.append({"family": "linear", "regularization": 1000.0 if task == "demand" else 0.1})
        boosted.append({"family": "boosted", "leaves": 63, "trees": 900,
                        "learning_rate": 0.025})
    return linear + boosted + bayes + svm + neural


def fit_svm(config, features, labels):
    """Exact kernel SVM for price direction on a fixed random subsample of at most max_rows
    training rows: exact kernel training costs roughly O(n^2) memory/time, so all ~70k
    half-hours per four years are not practical. gamma and C are hyperparameters chosen on
    validation data; training itself learns only the support-vector weights and offset."""
    columns = ([col for col in config["features"] if col in features]
               if config.get("features") else svm_columns(features))
    gamma = config["gamma"] / len(columns)              # gamma is relative to 1/#features
    rows = np.arange(len(features))
    max_rows = config.get("max_rows")
    if max_rows and len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(SEED).choice(len(rows), max_rows, replace=False))
    y = labels.to_numpy()[rows]
    if len(np.unique(y)) < 2:
        raise ValueError("SVM training rows need both classes")
    estimator = make_pipeline(StandardScaler(), SVC(
        kernel=config["kernel"], C=config["C"], gamma=gamma,
        degree=config.get("degree", 3), coef0=config.get("coef0", 0.0),
        class_weight="balanced", cache_size=1000))
    estimator.fit(svm_inputs(columns, features.iloc[rows]), y)
    return {"task": "price", "config": config, "features": columns,
            "estimator": estimator, "threshold": config.get("threshold", 0.5)}


def neural_net_loss_gradients(layers, x, y, pos_weight, masks=None):
    """One forward pass, the class-weighted cross-entropy loss, and its gradient with
    respect to every weight and bias by backpropagation.

    layers: [(W, b), ...] with W of shape (units_out, units_in); the last layer has 1 unit.
    masks:  one dropout mask per hidden layer (0 or 1/(1 - dropout)), or None for no dropout.
    Returns (loss, [(dL/dW, dL/db), ...]) in the same order as layers."""
    # Forward: keep every layer's output, because the backward pass needs them.
    outputs, h = [x], x
    for index, (weight, bias) in enumerate(layers[:-1]):
        h = np.maximum(h @ weight.T + bias, 0)            # linear step, then ReLU
        if masks is not None:
            h = h * masks[index]                           # dropout: zero random units
        outputs.append(h)
    weight, bias = layers[-1]
    logit = (h @ weight.T + bias)[:, 0]                    # one number per row
    # Loss: -[pos_weight * y * log(p) + (1 - y) * log(1 - p)], p = sigmoid(logit),
    # written with logaddexp so that large |logit| cannot overflow.
    row_weight = 1 + (pos_weight - 1) * y                  # pos_weight for "up" rows, else 1
    loss = np.mean(pos_weight * y * np.logaddexp(0, -logit) + (1 - y) * np.logaddexp(0, logit))
    probability = 1 / (1 + np.exp(-np.clip(logit, -60, 60)))
    # Backward: delta is dLoss/d(pre-activation) of the current layer, for every row.
    delta = (row_weight * (probability - y) / len(y))[:, None]
    gradients = []
    for index in reversed(range(len(layers))):
        gradients.append((delta.T @ outputs[index], delta.sum(axis=0)))
        if index > 0:
            delta = delta @ layers[index][0]               # chain rule through the linear step
            if masks is not None:
                delta = delta * masks[index - 1]           # ... through dropout
            delta = delta * (outputs[index] > 0)           # ... through ReLU (slope 1 or 0)
    return loss, gradients[::-1]


def fit_neural_net(config, features, labels):
    """Train the MLP ensemble by mini-batch gradient descent, in plain numpy.

    For each member (seed): start from small random weights, then repeat for a fixed number
    of epochs: shuffle the rows, and for every mini-batch (1) draw dropout masks,
    (2) forward pass and loss, (3) backpropagation, (4) one AdamW update, with the learning
    rate decaying to 0 along a cosine curve. The epoch budget is fixed, so nothing here
    depends on a validation year and the fit works for any training window."""
    log = config.get("log", "price")
    columns = neural_net_columns(features, log)
    params = {"columns": columns, "log_mask": neural_net_log_mask(columns, log),
              "clip_low": None, "clip_high": None}
    x = neural_net_transform(params, features)
    if config.get("winsorise"):
        # Cull outliers: bounds come from the training rows only and are stored, so
        # prediction applies exactly the same caps.
        params["clip_low"], params["clip_high"] = np.percentile(
            x, [config["winsorise"], 100 - config["winsorise"]], axis=0)
        x = np.clip(x, params["clip_low"], params["clip_high"])
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-8] = 1.0             # constant column: leave it at 0 after centring
    inputs = ((x - mean) / std).astype(np.float32)
    y = labels.to_numpy().astype(np.float32)
    rows = len(y)
    # Bagging: rows of each market day, so a member's resample keeps whole days together.
    day_codes = pd.factorize(market_days(features.index))[0]
    day_rows = np.split(np.argsort(day_codes, kind="stable"),
                        np.cumsum(np.bincount(day_codes))[:-1])
    batch = min(config["batch_size"], rows)
    # Weight "up" errors by n_down / n_up so both classes count equally in the loss,
    # matching balanced accuracy and keeping 0.5 a sensible default threshold.
    pos_weight = np.float32((y == 0).sum() / (y == 1).sum())
    keep = np.float32(1 - config["dropout"])
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8               # standard Adam constants
    sizes = [len(columns)] + list(config["hidden"]) + [1]
    members = []
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)                  # initial weights, order, dropout
        layers = []
        for units_in, units_out in zip(sizes[:-1], sizes[1:]):
            bound = 1 / np.sqrt(units_in)                  # PyTorch's default Linear init
            layers.append([rng.uniform(-bound, bound, (units_out, units_in)).astype(np.float32),
                           rng.uniform(-bound, bound, units_out).astype(np.float32)])
        first = [[np.zeros_like(p) for p in layer] for layer in layers]    # mean of gradients
        second = [[np.zeros_like(p) for p in layer] for layer in layers]   # mean of squares
        # Training rows for this member: all of them, or (bagging) as many days as the
        # window has, drawn with replacement, so some days appear twice and ~37% not at all.
        sample = (np.concatenate([day_rows[d] for d in rng.integers(0, len(day_rows),
                                                                     len(day_rows))])
                  if config.get("bootstrap") else np.arange(rows))
        steps_per_epoch = len(sample) // batch     # the ragged last batch is dropped
        total_steps = config["epochs"] * steps_per_epoch
        step = 0
        for _ in range(config["epochs"]):
            order = sample[rng.permutation(len(sample))]
            for start in range(0, steps_per_epoch * batch, batch):
                chosen = order[start:start + batch]
                masks = [(rng.random((batch, units), dtype=np.float32) < keep) / keep
                         for units in config["hidden"]]
                _, gradients = neural_net_loss_gradients(
                    layers, inputs[chosen], y[chosen], pos_weight, masks)
                rate = config["learning_rate"] * 0.5 * (1 + np.cos(np.pi * step / total_steps))
                step += 1
                for layer, grads, means, squares in zip(layers, gradients, first, second):
                    for k, gradient in enumerate(grads):
                        means[k] = beta1 * means[k] + (1 - beta1) * gradient
                        squares[k] = beta2 * squares[k] + (1 - beta2) * gradient * gradient
                        if k == 0:                         # weight decay on matrices, not biases
                            layer[k] *= 1 - rate * config["weight_decay"]
                        layer[k] -= (rate * (means[k] / (1 - beta1 ** step))
                                     / (np.sqrt(squares[k] / (1 - beta2 ** step)) + epsilon)
                                     ).astype(np.float32)
        members.append([(weight, bias) for weight, bias in layers])
    params.update(mean=mean, std=std, members=members, log=log,
                  winsorise=config.get("winsorise"))
    return {"task": "price", "config": config, "features": columns,
            "estimator": params, "threshold": config.get("threshold", 0.5)}


def fit_model(task, config, features, truth):
    # Keep the price-profile additions exclusive to classification.
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
    elif config["family"] == "naive_bayes":
        if task != "price":
            raise ValueError("Naive Bayes is only used for price direction")
        params = fit_naive_bayes(config, features, truth["price"])
        return {"task": task, "config": config,
                "features": params["discrete"] + params["continuous"],
                "estimator": params, "threshold": config.get("threshold", 0.5)}
    elif config["family"] == "svm":
        if task != "price":
            raise ValueError("The SVM is only used for price direction")
        return fit_svm(config, features, truth["price"])
    elif config["family"] == "neural_net":
        if task != "price":
            raise ValueError("The neural network is only used for price direction")
        return fit_neural_net(config, features, truth["price"])
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
    return {"task": task, "config": config, "features": columns,
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
                probabilities = price_probability(model, features.loc[valid])
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
    manifest["demand_blend"] = {"LightGBM": 0.5, "MLP": 0.5}
    manifest["mlp_epochs"] = 10
    with open(os.path.join(output, "demand_mlp_manifest.json"), encoding="utf-8") as handle:
        manifest["mlp_training"] = json.load(handle)
    manifest["inference"] = "LightGBM plus NumPy MLP; PyTorch used only in isolated training process"
    manifest["seconds"] = time.monotonic() - started
    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved artifacts to {output} in {manifest['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
