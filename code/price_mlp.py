"""Simple price-direction MLP (standalone experiment; does not touch the shared package).

Task: for each half-hour of day D+1, predict 1 if the NSW price is above the mean NSW price of
day D, else 0.  Protocol (same as Stage 1): fit on 2010-2013, select on 2014.

    python code/price_mlp.py                              # trains, prints a comparison table
    python code/price_mlp.py --output results/price_mlp   # also saves metrics.json

The network is a small feed-forward classifier on ~50 hand-picked features.  It is compared with
the "same half-hour yesterday" baseline, a logistic regression and a LightGBM classifier that use
exactly the same inputs, so any difference comes from the model and not the features.

PyTorch and LightGBM must not be loaded in the same process on macOS (their OpenMP runtimes
deadlock), so the reference models run in a child process of this script.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
from model import load_data, market_days, training_table  # noqa: E402

SEED = 9417

# A deliberately small, interpretable input set (names come from model.feature_frame).
FEATURES = (
    # where yesterday's / last week's price sat relative to the threshold (the target's reference)
    [f"price_margin_{lag}" for lag in (1, 2, 3, 7, 14, 21, 28)]
    + ["price_threshold", "nsw_price_previous_min", "nsw_price_previous_max",
       "nsw_price_previous_std", "nsw_price_mean_7d", "nsw_price_mean_28d",
       "nsw_price_recent_change", "nsw_price_weekly_mean",
       "vic_price_recent_change", "qld_price_recent_change"]
    # time of day / week / year
    + ["hour_sin", "hour_cos", "annual_sin", "annual_cos", "weekend", "holiday"]
    + [f"dow_{d}" for d in range(7)]
    # demand level and recent change
    + ["nsw_demand_lag_1", "nsw_demand_lag_7", "nsw_demand_previous_mean",
       "nsw_demand_recent_change", "vic_demand_recent_change", "qld_demand_recent_change"]
    # weather of the forecast day (known in advance)
    + ["syd_temp", "syd_heat", "syd_cool", "syd_solar", "syd_cloud", "mel_solar", "bne_solar",
       "syd_temp_today_max", "syd_solar_today_mean"]
)


def build_mlp(inputs, dropout=0.1):
    """inputs -> 64 -> 32 -> 1 logit; ReLU, light dropout."""
    from torch import nn
    return nn.Sequential(
        nn.Linear(inputs, 64), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(32, 1), nn.Flatten(0),
    )


def signed_log(frame, columns):
    """Price-valued columns contain spikes of thousands of $/MWh; log1p keeps the sign and
    squashes them so a single spike cannot dominate the scaling."""
    x = frame.loc[:, columns].to_numpy(np.float64, copy=True)
    for j, name in enumerate(columns):
        if "price" in name:
            x[:, j] = np.sign(x[:, j]) * np.log1p(np.abs(x[:, j]))
    return x


def best_threshold(truth, probability):
    grid = np.linspace(0.2, 0.8, 61)
    scores = [balanced_accuracy_score(truth, probability > t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def report(name, truth, probabilities=None, predictions=None, threshold=0.5):
    if predictions is None:
        predictions = probabilities > threshold
    return {"model": name, "threshold": threshold,
            "accuracy": accuracy_score(truth, predictions) * 100,
            "balanced_accuracy": balanced_accuracy_score(truth, predictions) * 100}


def prepare(data_path, train_until=2013, valid_from=2014, valid_to=2014):
    """Chronological split and scaled inputs. Default: train 2010-2013, validate 2014."""
    features, truth = training_table(load_data(data_path))
    years = market_days(features.index).year
    train, valid = years <= train_until, (years >= valid_from) & (years <= valid_to)
    columns = [c for c in FEATURES if c in features.columns]
    missing = sorted(set(FEATURES) - set(columns))
    if missing:
        print("WARNING: features not found, skipped:", missing, file=sys.stderr)
    scaler = StandardScaler().fit(signed_log(features.loc[train], columns))
    x_train = scaler.transform(signed_log(features.loc[train], columns)).astype(np.float32)
    x_valid = scaler.transform(signed_log(features.loc[valid], columns)).astype(np.float32)
    y_train = truth.loc[train, "price"].to_numpy()
    y_valid = truth.loc[valid, "price"].to_numpy()
    baseline = features.loc[valid, "price_margin_1"].to_numpy() > 0
    return columns, x_train, y_train, x_valid, y_valid, baseline


def fit_mlp(x_train, y_train, x_valid, y_valid, epochs=40, patience=6, lr=1e-3, decay=1e-4,
            fixed_epochs=None):
    """Adam on class-balanced binary cross-entropy; keep the epoch with the best validation
    balanced accuracy (at threshold 0.5), stop after `patience` epochs without improvement.

    With `fixed_epochs=N` the network trains for exactly N epochs and the validation labels are
    never used (no epoch selection), so the returned probabilities carry no tuning on them."""
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model = build_mlp(x_train.shape[1])
    positive = y_train.mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - positive) / positive,
                                                           dtype=torch.float32))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    data = torch.utils.data.TensorDataset(torch.from_numpy(x_train),
                                          torch.from_numpy(y_train.astype(np.float32)))
    loader = torch.utils.data.DataLoader(
        data, batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    valid = torch.from_numpy(x_valid)
    best, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []
    if fixed_epochs:
        epochs = fixed_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch_x)
        if fixed_epochs:
            history.append({"epoch": epoch, "train_loss": total / len(data)})
            print(f"  epoch {epoch:2d}  loss {total / len(data):.4f}", flush=True)
            continue
        model.eval()
        with torch.inference_mode():
            score = balanced_accuracy_score(y_valid, model(valid).numpy() > 0)
        history.append({"epoch": epoch, "train_loss": total / len(data), "valid_BA": score * 100})
        print(f"  epoch {epoch:2d}  loss {total / len(data):.4f}  valid BA {score * 100:.2f}",
              flush=True)
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if fixed_epochs:
        best_epoch = fixed_epochs
    else:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(valid)).numpy()
    return probabilities, best_epoch, history


def reference_models(data_path):
    """Logistic regression and LightGBM on the same scaled inputs (child process, no torch)."""
    from lightgbm import LGBMClassifier
    _, x_train, y_train, x_valid, y_valid, _ = prepare(data_path)
    logistic = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000,
                                  random_state=SEED)
    p_log = logistic.fit(x_train, y_train).predict_proba(x_valid)[:, 1]
    boosted = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             min_child_samples=160, class_weight="balanced", n_jobs=2,
                             verbosity=-1, random_state=SEED, deterministic=True)
    p_gbm = boosted.fit(x_train, y_train).predict_proba(x_valid)[:, 1]
    return [report("logistic regression (tuned threshold)", y_valid, p_log,
                   threshold=best_threshold(y_valid, p_log)),
            report("LightGBM (tuned threshold)", y_valid, p_gbm,
                   threshold=best_threshold(y_valid, p_gbm))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(
        ROOT, "data", "nem_weather_halfhourly_2010_2026.csv"))
    parser.add_argument("--output", help="directory for metrics.json (not written if omitted)")
    parser.add_argument("--reference-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.reference_only:   # child process: print JSON for the parent and exit
        print("JSON:" + json.dumps(reference_models(args.data)))
        return

    started = time.time()
    columns, x_train, y_train, x_valid, y_valid, baseline = prepare(args.data)
    print(f"{len(columns)} features; train {len(y_train):,} rows (2010-13), "
          f"validation {len(y_valid):,} rows (2014)")
    print(f"share of 'price up': train {y_train.mean():.3f}, validation {y_valid.mean():.3f}")

    rows = [report("baseline: same half-hour yesterday", y_valid, predictions=baseline)]
    print("\nTraining MLP")
    p_mlp, epoch, history = fit_mlp(x_train, y_train, x_valid, y_valid)
    rows.append(report("MLP (threshold 0.5)", y_valid, p_mlp))
    rows.append(report("MLP (tuned threshold)", y_valid, p_mlp,
                       threshold=best_threshold(y_valid, p_mlp)))

    print("\nRunning logistic regression and LightGBM in a separate process ...", flush=True)
    child = subprocess.run([sys.executable, os.path.abspath(__file__), "--reference-only",
                            "--data", args.data], capture_output=True, text=True, timeout=1800)
    lines = [line for line in child.stdout.splitlines() if line.startswith("JSON:")]
    if child.returncode != 0 or not lines:
        print("reference models failed:\n", child.stderr[-1500:])
    else:
        rows += json.loads(lines[-1][5:])

    print(f"\nBest MLP epoch: {epoch}   (total {time.time() - started:.0f}s)")
    print(f"\n{'model':44s} {'thr':>5s} {'acc %':>7s} {'bal.acc %':>10s}")
    for row in rows:
        print(f"{row['model']:44s} {row['threshold']:5.2f} {row['accuracy']:7.2f} "
              f"{row['balanced_accuracy']:10.2f}")
    print("\nNote: epoch and threshold are both chosen on 2014, so the tuned numbers are slightly "
          "optimistic for every model; this is a validation comparison, not a test score.")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump({"features": columns, "best_epoch": epoch, "results": rows,
                       "mlp_history": history}, handle, indent=2)
        print("saved", os.path.join(args.output, "metrics.json"))


if __name__ == "__main__":
    main()
