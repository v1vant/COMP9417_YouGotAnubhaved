"""Neural-net variants against the current configuration.

Questions: (1) does the signed log on raw prices help, or is it better to log only some
price features, or none, and instead cull outliers by clipping each input to its training
percentiles (winsorising); (2) does bagging (each ensemble member trained on its own
bootstrap resample of whole days) help; (3) does training for longer help;
(4) do more (or fewer) hidden layers help?

Protocol: fit on 2010-2013; each variant's decision threshold is tuned on 2014 and it is
scored on 2014. No later year is used.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
from model import NEURAL_NET, load_data, market_days, neural_net_probability, training_table
from train import fit_neural_net

THRESHOLDS = np.linspace(0.2, 0.8, 61)
VARIANTS = {
    "current (log all price features, 10 epochs)": {},
    "raw inputs (no log, no clipping)": {"log": "none"},
    "no log, clip 1-99%": {"log": "none", "winsorise": 1},
    "no log, clip 0.1-99.9%": {"log": "none", "winsorise": 0.1},
    "log price levels only, clip 1-99%": {"log": "levels", "winsorise": 1},
    "log price differences only, clip 1-99%": {"log": "differences", "winsorise": 1},
    "log all price features, clip 1-99%": {"winsorise": 1},
    "20 epochs": {"epochs": 20},
    "40 epochs": {"epochs": 40},
    "80 epochs": {"epochs": 80},
    "raw inputs, 40 epochs": {"log": "none", "epochs": 40},
    "bagging": {"bootstrap": True},
    "bagging, 40 epochs": {"bootstrap": True, "epochs": 40},
    "1 hidden layer (256)": {"hidden": [256]},
    "3 hidden layers (256-256-256)": {"hidden": [256, 256, 256]},
    "4 hidden layers (256-256-256-256)": {"hidden": [256, 256, 256, 256]},
    "3 hidden layers, funnel (512-256-128)": {"hidden": [512, 256, 128]},
    "3 hidden layers, funnel (256-128-64)": {"hidden": [256, 128, 64]},
    "4 hidden layers, 20 epochs": {"hidden": [256, 256, 256, 256], "epochs": 20},
}


def run(name, changes, train, tune_x):
    started = time.time()
    params = fit_neural_net(dict(NEURAL_NET, **changes), *train)["estimator"]
    return (name, neural_net_probability(params, train[0]), neural_net_probability(params, tune_x),
            time.time() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()
    started = time.time()

    features, truth = training_table(
        load_data(os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")))
    years = market_days(features.index).year
    train = (features.loc[years <= 2013], truth.loc[years <= 2013, "price"])
    tune_x, tune_y = features.loc[years == 2014], truth.loc[years == 2014, "price"].to_numpy()
    train_y = train[1].to_numpy()

    fitted = Parallel(n_jobs=args.jobs)(
        delayed(run)(name, changes, train, tune_x) for name, changes in VARIANTS.items())

    rows = []
    for name, train_p, tune_p, seconds in fitted:
        scores = [balanced_accuracy_score(tune_y, tune_p > t) for t in THRESHOLDS]
        threshold = float(THRESHOLDS[int(np.argmax(scores))])
        rows.append({"variant": name, "threshold": threshold,
                     "accuracy_2014": 100 * accuracy_score(tune_y, tune_p > threshold),
                     "BA_2014": 100 * max(scores),
                     "BA_training_2010_2013": 100 * balanced_accuracy_score(
                         train_y, train_p > threshold),
                     "fit_s": seconds})
    for row in rows:
        row["difference_vs_current"] = row["BA_2014"] - rows[0]["BA_2014"]

    table = pd.DataFrame(rows)
    table.round(3).to_csv(os.path.join(args.output, "neural_net_variants.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.round(2).to_string(index=False))
    print(f"Saved neural_net_variants.csv in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
