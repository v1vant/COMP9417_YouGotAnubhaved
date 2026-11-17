"""How the RBF SVM responds to different treatments of outlying input values.

Every variant starts from RAW inputs (raw $/MWh price lags instead of the log lags) and
uses the same SVM: RBF kernel, gamma = 1/#features, C = 0.3, class_weight="balanced",
the same fixed random training subsample from 2010-2013, the decision threshold tuned on
2014. "signed_log" reproduces the production SVM. No later year is used.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
from model import SVM, load_data, market_days, svm_columns, training_table

SEED = 9417
THRESHOLDS = np.linspace(0.2, 0.8, 61)
VARIANTS = ("none", "signed_log", "winsorise_1_99", "winsorise_0.1_99.9", "robust_scaling",
            "quantile_normal", "signed_log+winsorise_1_99", "remove_outlier_rows")


def signed_log(x, price):
    x = x.copy()
    x[:, price] = np.sign(x[:, price]) * np.log1p(np.abs(x[:, price]))
    return x


def winsorise(train, others, low, high):
    lo, hi = np.percentile(train, [low, high], axis=0)
    return [np.clip(x, lo, hi) for x in [train] + others]


def prepare(variant, train, others, price):
    """Return (transformed train, transformed others, scaler). Fitted on train only."""
    scaler = StandardScaler()
    if variant in ("none", "remove_outlier_rows"):
        pass
    elif variant == "signed_log":
        train, others = signed_log(train, price), [signed_log(x, price) for x in others]
    elif variant.startswith("winsorise"):
        low, high = (float(v) for v in variant.split("_")[1:3])
        train, *others = winsorise(train, others, low, high)
    elif variant == "robust_scaling":
        scaler = RobustScaler()
    elif variant == "quantile_normal":
        scaler = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                     subsample=200_000, random_state=SEED)
    elif variant == "signed_log+winsorise_1_99":
        train, others = signed_log(train, price), [signed_log(x, price) for x in others]
        train, *others = winsorise(train, others, 1, 99)
    return train, others, scaler


def run(variant, x_train, y_train, x_eval, y_eval, price, rows, rng_seed):
    started = time.time()
    dropped = 0.0
    if variant == "remove_outlier_rows":
        lo, hi = np.percentile(x_train[:, price], [0.1, 99.9], axis=0)
        inside = ((x_train[:, price] >= lo) & (x_train[:, price] <= hi)).all(axis=1)
        dropped = 1 - inside.mean()
        pool = np.flatnonzero(inside)
        rows = np.sort(np.random.default_rng(rng_seed).choice(pool, min(len(rows), len(pool)),
                                                              replace=False))
    train, evals, scaler = prepare(variant, x_train[rows], list(x_eval.values()), price)
    evals = dict(zip(x_eval, evals))
    svc = SVC(kernel="rbf", gamma=SVM["gamma"] / x_train.shape[1], C=SVM["C"],
              class_weight="balanced", cache_size=1000)
    model = make_pipeline(scaler, svc).fit(train, y_train[rows])
    scores = {year: 1 / (1 + np.exp(-model.decision_function(x))) for year, x in evals.items()}
    first = min(scores)
    tuned = [balanced_accuracy_score(y_eval[first], scores[first] > t) for t in THRESHOLDS]
    threshold = float(THRESHOLDS[int(np.argmax(tuned))])
    scaled = model[0].transform(train)
    result = {"variant": variant, "threshold": threshold,
              f"BA_{first}_default": 100 * balanced_accuracy_score(y_eval[first], scores[first] > 0.5),
              f"BA_{first}_tuned": 100 * max(tuned)}
    for year in sorted(scores)[1:]:
        result[f"BA_{year}"] = 100 * balanced_accuracy_score(y_eval[year], scores[year] > threshold)
    result.update({"support_vectors": int(model[-1].n_support_.sum()),
                   "max_abs_scaled_input": float(np.abs(scaled).max()),
                   "train_rows_dropped_%": 100 * dropped, "fit_s": time.time() - started})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-year", type=int, default=2014)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    features, truth = training_table(
        load_data(os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")))
    # Raw inputs: the production column set with log price lags swapped for raw lags.
    columns = [col.replace("_price_loglag_", "_price_lag_") for col in svm_columns(features)]
    price = np.array(["price" in col for col in columns])
    years = market_days(features.index).year
    train_mask = years < args.select_year
    x_train = features.loc[train_mask, columns].to_numpy(np.float64)
    y_train = truth.loc[train_mask, "price"].to_numpy()
    x_eval = {y: features.loc[years == y, columns].to_numpy(np.float64)
              for y in (args.select_year,)}
    y_eval = {y: truth.loc[years == y, "price"].to_numpy() for y in x_eval}
    rows = np.sort(np.random.default_rng(SEED).choice(len(x_train), SVM["max_rows"],
                                                      replace=False))
    print(f"{len(columns)} raw inputs ({price.sum()} price-based); "
          f"max |raw price input| in training: {np.abs(x_train[:, price]).max():,.0f}", flush=True)

    results = Parallel(n_jobs=args.jobs)(
        delayed(run)(v, x_train, y_train, x_eval, y_eval, price, rows, SEED) for v in VARIANTS)
    table = pd.DataFrame(results).sort_values(f"BA_{args.select_year}_tuned", ascending=False)
    table.round(3).to_csv(os.path.join(args.output, "svm_outlier_comparison.csv"), index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
