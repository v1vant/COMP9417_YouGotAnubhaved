"""RBF-SVM feature pruning by permutation importance, then hyperparameter re-tuning.

An RBF SVM has no per-feature weights, so a feature's weight is measured as the balanced
accuracy lost when that feature is shuffled in held-out data (permutation importance).

Default split: fit on 2010-2012, make every choice on 2013, test on 2014.

1. Backward elimination: fit on all SVM features; each round, drop the --drop-frac least
   important features (by permutation on the selection year), refit, and repeat down to
   --min-features. Gradual removal matters: correlated features hide each other's
   importance, so removing every low scorer at once could delete all copies of a signal.
2. Keep the smallest feature set whose selection-year BA is within --tolerance of the best.
3. Re-tune gamma and C for that set on the selection year (gamma is a hyperparameter, not
   learned by training, and the number of features it is scaled by has changed).

The test year is reported for every stage but never used for any choice.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
from model import SVM, load_data, market_days, svm_columns, svm_inputs, training_table

SEED = 9417
THRESHOLDS = np.linspace(0.2, 0.8, 61)


def fit(columns, train, rows, gamma, C):
    x = svm_inputs(columns, train[0].iloc[rows])
    y = train[1]["price"].to_numpy()[rows]
    svc = SVC(kernel="rbf", gamma=gamma / len(columns), C=C, class_weight="balanced",
              cache_size=1000)
    return make_pipeline(StandardScaler(), svc).fit(x, y)


def ba(model, x, y, threshold):
    probability = 1 / (1 + np.exp(-model.decision_function(x)))
    return 100 * balanced_accuracy_score(y, probability > threshold)


def tuned(model, x, y):
    probability = 1 / (1 + np.exp(-model.decision_function(x)))
    scores = [balanced_accuracy_score(y, probability > t) for t in THRESHOLDS]
    best = int(np.argmax(scores))
    return 100 * scores[best], float(THRESHOLDS[best])


def permutation_drop(model, x, y, threshold, base, column, seed):
    shuffled = x.copy()
    shuffled[:, column] = np.random.default_rng(seed).permutation(shuffled[:, column])
    return base - ba(model, shuffled, y, threshold)


def evaluate(columns, data, rows, gamma, C):
    model = fit(columns, data["train"], rows, gamma, C)
    x_sel = svm_inputs(columns, data["select"][0])
    y_sel = data["select"][1]["price"].to_numpy()
    sel, threshold = tuned(model, x_sel, y_sel)
    test = ba(model, svm_inputs(columns, data["test"][0]),
              data["test"][1]["price"].to_numpy(), threshold)
    return model, x_sel, y_sel, sel, test, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-end", type=int, default=2012, help="last training year")
    parser.add_argument("--select-year", type=int, default=2013)
    parser.add_argument("--test-year", type=int, default=2014)
    parser.add_argument("--train-rows", type=int, default=10000, help="SVM training subsample")
    parser.add_argument("--perm-rows", type=int, default=5000,
                        help="selection-year rows used for permutation importance")
    parser.add_argument("--drop-frac", type=float, default=0.10)
    parser.add_argument("--min-features", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.1, help="BA points")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--output", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()
    if not args.train_end < args.select_year < args.test_year:
        raise ValueError("Years must be chronological: train < select < test")

    features, truth = training_table(
        load_data(os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")))
    years = market_days(features.index).year
    data = {key: (features.loc[mask], truth.loc[mask]) for key, mask in
            (("train", years <= args.train_end), ("select", years == args.select_year),
             ("test", years == args.test_year))}
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(len(data["train"][0]),
                              min(args.train_rows, len(data["train"][0])), replace=False))
    perm = np.sort(rng.choice(len(data["select"][0]),
                              min(args.perm_rows, len(data["select"][0])), replace=False))
    gamma, C = SVM["gamma"], SVM["C"]
    split = f"select {args.select_year} / test {args.test_year}"

    # 1. Backward elimination by permutation importance.
    columns, rounds = svm_columns(features), []
    while True:
        started = time.time()
        model, x_sel, y_sel, sel, test, threshold = evaluate(columns, data, rows, gamma, C)
        record = {"features": len(columns), "BA_select": round(sel, 2),
                  "BA_test": round(test, 2), "threshold": threshold, "columns": list(columns)}
        if len(columns) > args.min_features:
            base = ba(model, x_sel[perm], y_sel[perm], threshold)
            drops = np.array(Parallel(n_jobs=args.jobs)(
                delayed(permutation_drop)(model, x_sel[perm], y_sel[perm], threshold, base,
                                          j, SEED + j) for j in range(len(columns))))
            k = min(max(1, int(round(args.drop_frac * len(columns)))),
                    len(columns) - args.min_features)
            weakest = np.argsort(drops)[:k]
            record["removed"] = {columns[j]: round(float(drops[j]), 3) for j in weakest}
        rounds.append(record)
        print(f"{len(columns):4d} features: BA {split} = {sel:.2f} / {test:.2f} "
              f"({time.time() - started:.0f}s)", flush=True)
        if "removed" not in record:
            break
        columns = [col for col in columns if col not in record["removed"]]

    # 2. Smallest set within tolerance of the best selection-year score.
    best = max(r["BA_select"] for r in rounds)
    chosen = min((r for r in rounds if r["BA_select"] >= best - args.tolerance),
                 key=lambda r: r["features"])
    columns = chosen["columns"]
    print(f"\nChosen: {len(columns)} features (best selection BA {best:.2f}, "
          f"chosen {chosen['BA_select']:.2f})", flush=True)

    # 3. Re-tune gamma and C for the chosen set on the selection year.
    grid = [(g, c) for g in (0.3, 1.0, 3.0) for c in (0.1, 0.3, 1.0)]
    results = Parallel(n_jobs=args.jobs)(
        delayed(lambda g, c: (g, c) + evaluate(columns, data, rows, g, c)[3:])(g, c)
        for g, c in grid)
    tuning = [{"gamma": g, "C": c, "BA_select": round(s, 2), "BA_test": round(t, 2),
               "threshold": th} for g, c, s, t, th in results]
    best_params = max(tuning, key=lambda r: r["BA_select"])
    print("\nHyperparameters for the chosen set (gamma in units of 1/#features):")
    for row in sorted(tuning, key=lambda r: -r["BA_select"]):
        print(f"  gamma={row['gamma']:<4} C={row['C']:<4} BA {split} = "
              f"{row['BA_select']:.2f} / {row['BA_test']:.2f}")

    # Final importance of each kept feature, for the report.
    model, x_sel, y_sel, sel, test, threshold = evaluate(
        columns, data, rows, best_params["gamma"], best_params["C"])
    base = ba(model, x_sel[perm], y_sel[perm], threshold)
    drops = Parallel(n_jobs=args.jobs)(
        delayed(permutation_drop)(model, x_sel[perm], y_sel[perm], threshold, base, j, SEED + j)
        for j in range(len(columns)))
    importance = sorted(zip(columns, drops), key=lambda item: -item[1])
    result = {
        "settings": vars(args) | {"output": None},
        "all_features": {k: rounds[0][k] for k in ("features", "BA_select", "BA_test")},
        "chosen": {"features": columns, "gamma": best_params["gamma"], "C": best_params["C"],
                   "BA_select": round(sel, 2), "BA_test": round(test, 2)},
        "importance_points": {name: round(float(drop), 3) for name, drop in importance},
        "tuning": tuning,
        "rounds": [{k: v for k, v in r.items() if k != "columns"} for r in rounds],
    }
    with open(os.path.join(args.output, "svm_feature_selection.json"), "w",
              encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nAll {rounds[0]['features']} features: {rounds[0]['BA_select']:.2f} / "
          f"{rounds[0]['BA_test']:.2f};  chosen {len(columns)} with gamma="
          f"{best_params['gamma']}, C={best_params['C']}: {sel:.2f} / {test:.2f}")
    print("\nImportance of kept features (BA points lost when shuffled):")
    for name, drop in importance:
        print(f"  {drop:6.3f}  {name}")


if __name__ == "__main__":
    main()
