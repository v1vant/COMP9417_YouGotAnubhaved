"""Evidence behind the neural-network design (price direction), reproducible in numpy.

Protocol (as in the project spec): fit on 2010-2013; every choice, including the decision
threshold, is made on 2014. No later year is used.

1. How much fitting: epochs x dropout x learning rate, with training and validation
   balanced accuracy, to show where overfitting starts.
2. Network size at the chosen recipe.
3. Ensemble size: single networks versus the mean of 3 and 5 seeds.
4. The production configuration (model.NEURAL_NET): accuracy and balanced accuracy.

One factor is varied at a time around model.NEURAL_NET; nothing here feeds back into the
package automatically.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
from model import NEURAL_NET, load_data, market_days, neural_net_probability, training_table
from train import fit_neural_net

THRESHOLDS = np.linspace(0.2, 0.8, 61)


def balanced(truth, probability, threshold):
    return 100 * balanced_accuracy_score(truth, probability > threshold)


def tuned(truth, probability):
    scores = [balanced(truth, probability, t) for t in THRESHOLDS]
    return max(scores), float(THRESHOLDS[int(np.argmax(scores))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(ROOT, "results"))
    parser.add_argument("--quick", action="store_true", help="tiny grid, for a smoke test")
    args = parser.parse_args()
    started = time.time()

    features, truth = training_table(
        load_data(os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")))
    years = market_days(features.index).year
    part = {name: (features.loc[mask], truth.loc[mask, "price"]) for name, mask in
            (("fit", years <= 2013), ("tune", years == 2014))}
    rows = []

    def search(stage, label, seeds=(0,), **changes):
        """Fit on 2010-2013; report training BA (at 0.5) and 2014 BA (threshold tuned on 2014)."""
        config = dict(NEURAL_NET, seeds=list(seeds), **changes)
        fit_started = time.time()
        params = fit_neural_net(config, *part["fit"])["estimator"]
        train_x, train_y = part["fit"]
        val_x, val_y = part["tune"]
        val_probability = neural_net_probability(params, val_x)
        score, threshold = tuned(val_y.to_numpy(), val_probability)
        row = {"stage": stage, "setting": label, "seeds": len(seeds),
               "BA_train_at_0.5": balanced(train_y.to_numpy(),
                                           neural_net_probability(params, train_x), 0.5),
               "BA_2014_at_0.5": balanced(val_y.to_numpy(), val_probability, 0.5),
               "BA_2014_tuned": score, "threshold": threshold,
               "fit_s": time.time() - fit_started}
        rows.append(row)
        print(f"[{stage}] {label}: train {row['BA_train_at_0.5']:.2f} | 2014 {score:.2f}",
              flush=True)
        return params

    # 1. How much fitting.
    epochs_grid = (2, 10) if args.quick else (2, 5, 10, 20, 40)
    for dropout in ((0.5,) if args.quick else (0.0, 0.2, 0.5)):
        for rate in ((3e-4,) if args.quick else (3e-4, 1e-3)):
            for epochs in epochs_grid:
                search("fitting", f"dropout={dropout} lr={rate:g} epochs={epochs}",
                       dropout=dropout, learning_rate=rate, epochs=epochs)

    # 2. Network size at the chosen recipe.
    for hidden in ([[64, 64]] if args.quick else [[64], [64, 64], [256, 256], [512, 512],
                                                  [256, 256, 256]]):
        search("size", f"hidden={hidden}", hidden=hidden)

    # 3. Ensemble size: fit 10 members once, then average subsets of them.
    count = 4 if args.quick else 10
    params = search("ensemble", f"{count} members together", seeds=range(count))
    val_x, val_y = part["tune"]
    each = np.array([neural_net_probability(dict(params, members=[m]), val_x)
                     for m in params["members"]])
    for size in ((1, 2) if args.quick else (1, 3, 5)):
        groups = [range(start, start + size) for start in range(0, count - size + 1, size)]
        scores = [tuned(val_y.to_numpy(), each[list(g)].mean(axis=0))[0] for g in groups]
        rows.append({"stage": "ensemble", "setting": f"mean of {size} seed(s), "
                     f"{len(groups)} disjoint group(s)", "seeds": size,
                     "BA_2014_tuned": float(np.mean(scores)),
                     "BA_2014_std_across_groups": float(np.std(scores))})
        print(f"[ensemble] {size} seed(s): 2014 {np.mean(scores):.2f} +/- {np.std(scores):.2f}",
              flush=True)

    # 4. The production configuration: threshold tuned on 2014.
    config = dict(NEURAL_NET, **({"seeds": [0, 1], "epochs": 3} if args.quick else {}))
    fit_started = time.time()
    params = fit_neural_net(config, *part["fit"])["estimator"]
    fit_seconds = time.time() - fit_started
    row = {"stage": "production", "setting": "model.NEURAL_NET, fit 2010-2013",
           "seeds": len(config["seeds"]), "fit_s": fit_seconds}
    probability = {name: neural_net_probability(params, part[name][0])
                   for name in ("fit", "tune")}
    truth_of = {name: part[name][1].to_numpy() for name in probability}
    _, threshold = tuned(truth_of["tune"], probability["tune"])   # the only tuning: on 2014
    row["threshold"] = threshold
    for name, year in (("fit", "2010-2013"), ("tune", "2014")):
        predicted = probability[name] > threshold
        row[f"accuracy_{year}"] = 100 * accuracy_score(truth_of[name], predicted)
        row[f"BA_{year}"] = 100 * balanced_accuracy_score(truth_of[name], predicted)
        row[f"share_up_{year}"] = 100 * truth_of[name].mean()
    rows.append(row)
    print(f"[production] threshold {threshold:.2f} (tuned on 2014) | accuracy / BA: "
          f"training 2010-2013 {row['accuracy_2010-2013']:.2f} / {row['BA_2010-2013']:.2f} | "
          f"2014 {row['accuracy_2014']:.2f} / {row['BA_2014']:.2f} | fit {fit_seconds:.0f}s",
          flush=True)

    name = "neural_net_experiments_quick.csv" if args.quick else "neural_net_experiments.csv"
    pd.DataFrame(rows).round(3).to_csv(os.path.join(args.output, name), index=False)
    print(f"Saved {name} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
