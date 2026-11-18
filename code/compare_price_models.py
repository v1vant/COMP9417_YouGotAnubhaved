"""Compare price-direction (classification) models under one protocol: fit 2010-2013, select on 2014.

    python code/compare_price_models.py --list                      # show what can be run
    python code/compare_price_models.py --quick                     # everything, with small models
    python code/compare_price_models.py --models mlp svm            # chosen model types or exact names
    python code/compare_price_models.py --models compact            # same-45-input comparison, ~15 s
    python code/compare_price_models.py --models all --output results/model_comparison

Model names
  native   the team's own price models, each with the inputs train.py gives it
             linear_0, linear_1      logistic regression (team features, no price-margin columns)
             boosted_0, boosted_1    LightGBM classifier
             naive_bayes_0           mixed Naive Bayes
             svm_0                   RBF-kernel SVM on a fixed subsample
  compact  models that share exactly the same 45 hand-picked inputs, so differences are due to
           the model alone
             compact_logistic, compact_boosted, mlp
Groups: all, native, compact.  A family name (e.g. `linear`) selects every model of that family.

Every model is scored on the same 2014 half-hours with accuracy and balanced accuracy, at the
decision threshold that maximises 2014 balanced accuracy (as select_models does) and at 0.5.
Both baselines are always included.  Because the threshold and the MLP epoch are chosen on 2014,
all scores are validation numbers and slightly optimistic; they are not test scores.

PyTorch and LightGBM must not share a process on macOS (their OpenMP runtimes deadlock), so models
that need PyTorch run in a child process of this script.

To add a model: write a `run_<name>(data, config)` returning (probabilities, n_inputs, note)
and add an Entry in `torch_entries()` or `native_entries()` / `compact_entries()`.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
sys.path.insert(0, HERE)
import price_mlp as pm  # noqa: E402  (imports no torch / lightgbm at module level)
from model import load_data, market_days, price_probability, training_table  # noqa: E402

SEED = 9417
DEFAULT_DATA = os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")


@dataclass
class Entry:
    name: str
    family: str
    group: str                 # "native" or "compact"
    needs_torch: bool
    description: str
    run: Callable              # run(data, config) -> (probabilities, n_inputs, note)
    config: dict = None


class Data:
    """Chronological split built lazily: train 2010-2013, validate 2014."""

    def __init__(self, path, quick=False):
        self.path, self.quick = path, quick
        self._full = self._compact = None

    def full(self):
        if self._full is None:
            features, truth = training_table(load_data(self.path))
            years = market_days(features.index).year
            train, valid = years <= 2013, years == 2014
            self._full = (features.loc[train], truth.loc[train],
                          features.loc[valid], truth.loc[valid])
        return self._full

    def compact(self):
        if self._compact is None:
            self._compact = pm.prepare(self.path)
        return self._compact

    def y_valid(self):
        if self._full is not None:
            return self._full[3]["price"].to_numpy()
        if self._compact is not None:
            return self._compact[4]
        return self.full()[3]["price"].to_numpy()


# ---- runners -----------------------------------------------------------------------------
def run_native(data, config):
    from train import fit_model          # imports LightGBM: never call this in the torch child
    x_train, t_train, x_valid, _ = data.full()
    bundle = fit_model("price", config, x_train, t_train)
    return price_probability(bundle, x_valid), len(bundle["features"]), ""


def run_compact_logistic(data, config):
    _, x_train, y_train, x_valid, _, _ = data.compact()
    model = LogisticRegression(C=config["C"], class_weight="balanced", max_iter=1000,
                               random_state=SEED).fit(x_train, y_train)
    return model.predict_proba(x_valid)[:, 1], x_train.shape[1], ""


def run_compact_boosted(data, config):
    from lightgbm import LGBMClassifier
    _, x_train, y_train, x_valid, _, _ = data.compact()
    trees = 100 if data.quick else config["trees"]
    model = LGBMClassifier(n_estimators=trees, learning_rate=0.05, num_leaves=31,
                           min_child_samples=160, class_weight="balanced", n_jobs=2,
                           verbosity=-1, random_state=SEED, deterministic=True)
    return model.fit(x_train, y_train).predict_proba(x_valid)[:, 1], x_train.shape[1], ""


def run_mlp(data, config):
    _, x_train, y_train, x_valid, y_valid, _ = data.compact()
    probabilities, epoch, _ = pm.fit_mlp(x_train, y_train, x_valid, y_valid,
                                         epochs=10 if data.quick else config["epochs"],
                                         patience=config["patience"], decay=config["decay"])
    return probabilities, x_train.shape[1], f"best epoch {epoch}"


# ---- registry ----------------------------------------------------------------------------
def describe(config):
    family = config["family"]
    if family == "linear":
        return f"logistic regression, C={config['regularization']}"
    if family == "boosted":
        return f"LightGBM, {config['leaves']} leaves, {config['trees']} trees"
    if family == "naive_bayes":
        return f"mixed Naive Bayes, {len(config['continuous']) + len(config['discrete'])} inputs"
    return f"RBF SVM, C={config['C']}, {config.get('max_rows')} training rows"


def native_entries(quick):
    from train import candidates          # imports LightGBM: skipped in the torch child
    entries, seen = [], {}
    for config in candidates("price", quick):
        family = config["family"]
        index = seen[family] = seen.get(family, -1) + 1
        entries.append(Entry(f"{family}_{index}", family, "native", False,
                             describe(config), run_native, config))
    return entries


def compact_entries():
    return [
        Entry("compact_logistic", "compact_logistic", "compact", False,
              "logistic regression, C=0.1, 45 shared inputs", run_compact_logistic, {"C": 0.1}),
        Entry("compact_boosted", "compact_boosted", "compact", False,
              "LightGBM, 31 leaves, 300 trees, 45 shared inputs", run_compact_boosted,
              {"trees": 300}),
    ] + torch_entries()


def torch_entries():
    return [Entry("mlp", "mlp", "compact", True,
                  "MLP 45-64-32-1, dropout 0.1, early stopping, 45 shared inputs", run_mlp,
                  {"epochs": 40, "patience": 6, "decay": 1e-4})]


def registry(quick):
    return native_entries(quick) + compact_entries()


def select(entries, tokens):
    names = {e.name for e in entries}
    chosen = []
    for token in tokens:
        if token == "all":
            matched = entries
        elif token in ("native", "compact"):
            matched = [e for e in entries if e.group == token]
        else:
            matched = [e for e in entries if token in (e.name, e.family)]
        if not matched:
            raise SystemExit(f"Unknown model {token!r}. Available: {sorted(names)} "
                             "or groups all/native/compact (see --list)")
        chosen += [e for e in matched if e not in chosen]
    return chosen


# ---- scoring -----------------------------------------------------------------------------
def score(name, family, inputs, probabilities, y, seconds, note=""):
    threshold = pm.best_threshold(y, probabilities)
    predicted = probabilities > threshold
    return {"model": name, "family": family, "inputs": inputs, "threshold": threshold,
            "accuracy": accuracy_score(y, predicted) * 100,
            "balanced_accuracy": balanced_accuracy_score(y, predicted) * 100,
            "balanced_accuracy_at_0.5": balanced_accuracy_score(y, probabilities > 0.5) * 100,
            "fit_seconds": seconds, "note": note}


def baseline_rows(data):
    x_valid, y = data.full()[2], data.y_valid()
    rows = []
    for name, column in (("baseline: yesterday", "price_margin_1"),
                         ("baseline: last week", "price_margin_7")):
        predicted = x_valid[column].to_numpy() > 0
        rows.append({"model": name, "family": "baseline", "inputs": 0, "threshold": np.nan,
                     "accuracy": accuracy_score(y, predicted) * 100,
                     "balanced_accuracy": balanced_accuracy_score(y, predicted) * 100,
                     "balanced_accuracy_at_0.5": np.nan, "fit_seconds": 0.0, "note": ""})
    return rows


def run_entries(entries, data):
    rows, y = [], None
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry.name}: {entry.description}", flush=True)
        started = time.time()
        try:
            probabilities, inputs, note = entry.run(data, entry.config)
            y = data.y_valid() if y is None else y
            row = score(entry.name, entry.family, inputs, probabilities, y,
                        time.time() - started, note)
            print(f"    done in {row['fit_seconds']:.0f}s   balanced accuracy "
                  f"{row['balanced_accuracy']:.2f}%", flush=True)
        except Exception as error:   # keep going: one failing model should not lose the rest
            row = {"model": entry.name, "family": entry.family, "inputs": np.nan,
                   "threshold": np.nan, "accuracy": np.nan, "balanced_accuracy": np.nan,
                   "balanced_accuracy_at_0.5": np.nan, "fit_seconds": time.time() - started,
                   "note": f"FAILED: {type(error).__name__}: {error}"}
            print(f"    {row['note']}", flush=True)
        rows.append(row)
    return rows


def print_table(rows):
    models = sorted([r for r in rows if r["family"] != "baseline"],
                    key=lambda r: -np.nan_to_num(r["balanced_accuracy"], nan=-1))
    baselines = [r for r in rows if r["family"] == "baseline"]
    header = f"{'model':20s} {'inputs':>6s} {'thr':>5s} {'acc %':>7s} {'bal.acc %':>10s} " \
             f"{'@0.5':>7s} {'secs':>6s}  note"
    print("\n" + header + "\n" + "-" * len(header))
    for r in models + baselines:
        fmt = lambda v, f: "" if pd.isna(v) else format(v, f)  # noqa: E731
        print(f"{r['model']:20s} {fmt(r['inputs'], '.0f'):>6s} {fmt(r['threshold'], '.2f'):>5s} "
              f"{fmt(r['accuracy'], '.2f'):>7s} {fmt(r['balanced_accuracy'], '.2f'):>10s} "
              f"{fmt(r['balanced_accuracy_at_0.5'], '.2f'):>7s} {r['fit_seconds']:6.0f}  "
              f"{r['note']}")
    print("\nSplit: fit 2010-2013, validate 2014. Thresholds (and the MLP epoch) are chosen on "
          "2014, so these\nare validation scores, slightly optimistic, not test scores. "
          "'inputs' differs between native and\ncompact models; compare compact models with each "
          "other for a like-for-like result.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="model names, family names or groups (default: all)")
    parser.add_argument("--list", action="store_true", help="list available models and exit")
    parser.add_argument("--quick", action="store_true",
                        help="smaller/faster models (smoke test, not final accuracy)")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", help="directory to save comparison.csv")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json-out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:   # child process: only the PyTorch models, no LightGBM import
        entries = [e for e in torch_entries() if e.name in args.models]
        rows = run_entries(entries, Data(args.data, args.quick))
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        return

    entries = registry(args.quick)
    if args.list:
        print(f"{'name':18s} {'group':8s} {'process':8s} description")
        for e in entries:
            print(f"{e.name:18s} {e.group:8s} {'torch' if e.needs_torch else 'main':8s} "
                  f"{e.description}")
        print("\nGroups: all, native, compact. A family name selects every model of that family."
              "\nNative = team models with their own inputs; compact = shared 45 inputs.")
        return

    chosen = select(entries, args.models)
    print(f"Running {len(chosen)} model(s): {', '.join(e.name for e in chosen)}"
          f"{'  [quick]' if args.quick else ''}\n", flush=True)
    data = Data(args.data, args.quick)
    rows = run_entries([e for e in chosen if not e.needs_torch], data)

    torch_names = [e.name for e in chosen if e.needs_torch]
    if torch_names:
        print("\nStarting a separate process for PyTorch models ...", flush=True)
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "rows.json")
            command = [sys.executable, os.path.abspath(__file__), "--worker", "--json-out", out,
                       "--data", args.data] + (["--quick"] if args.quick else []) \
                + ["--models"] + torch_names
            result = subprocess.run(command)
            if result.returncode == 0 and os.path.exists(out):
                with open(out, encoding="utf-8") as handle:
                    rows += json.load(handle)
            else:
                rows += [{"model": n, "family": "mlp", "inputs": np.nan, "threshold": np.nan,
                          "accuracy": np.nan, "balanced_accuracy": np.nan,
                          "balanced_accuracy_at_0.5": np.nan, "fit_seconds": 0.0,
                          "note": f"FAILED: worker exit code {result.returncode}"}
                         for n in torch_names]

    rows += baseline_rows(data)
    print_table(rows)
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(args.output, "comparison.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print("saved", path)


if __name__ == "__main__":
    main()
