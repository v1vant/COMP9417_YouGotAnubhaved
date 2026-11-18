"""Compare price-direction (classification) models under one protocol: fit 2010-2013, select on 2014.

    python code/compare_price_models.py --list                      # show what can be run
    python code/compare_price_models.py --quick                     # everything, with small models
    python code/compare_price_models.py --models mlp svm            # chosen model types or exact names
    python code/compare_price_models.py --models compact            # the 45-input reference models
    python code/compare_price_models.py --models all --output results/model_comparison
    python code/compare_price_models.py --holdout --confirm-holdout --models mlp boosted_0

Model names
  native   the team's own price models, each with the inputs train.py gives it
             linear_0, linear_1      logistic regression (team features, no price-margin columns)
             boosted_0, boosted_1    LightGBM classifier
             naive_bayes_0           mixed Naive Bayes
             svm_0                   RBF-kernel SVM on a fixed subsample
  compact  reference models on the same 45 hand-picked inputs (a small, interpretable set)
             compact_logistic, compact_boosted
  mlp      the price MLP: a small neural network on all ~535 non-duplicate inputs, like the
           demand MLP that sees all its 405 features (code and settings live in price_mlp.py)
  blend    0.5 * strongest native LightGBM + 0.5 * mlp (probabilities averaged, then the usual
           threshold search), the price counterpart of the demand blend.  Runs the models it
           needs automatically.
Groups: all, native, compact, blend.  A family name (e.g. `linear`) selects every model of that
family.

Every model is scored on the same 2014 half-hours with accuracy and balanced accuracy, at the
decision threshold that maximises 2014 balanced accuracy (as select_models does) and at 0.5.
Both baselines are always included.  Because the threshold and the MLP epoch are chosen on 2014,
all scores are validation numbers and slightly optimistic; they are not test scores.

HOLDOUT MODE (--holdout --confirm-holdout): one-off check on 2025-26.  Models are fitted on every
year up to 2024 and scored on 2025-26, the closest analogue of the live week.  Nothing may be
tuned on the holdout, so the threshold is fixed at 0.5 (all models use balanced class weights),
the MLP trains a fixed 6 epochs (its best epoch on 2014) and each score comes with a 95%
day-block bootstrap interval.  Do not choose or tune models by this result (that would turn the
holdout into a selection set), and say in the report that the holdout was used as a final
check.  The saved package artifacts cannot be used for this: they were trained through Oct 2026.
Native models here use threshold 0.5, not the tuned threshold of the deployed models, so numbers
can differ slightly from the team's holdout results.

PyTorch and LightGBM must not share a process on macOS (their OpenMP runtimes deadlock), so models
that need PyTorch run in a child process that hands its probabilities back; the blend is
computed in the main process.

To add a model: write a `run_<name>(data, config)` returning (probabilities, n_inputs, note)
and add an Entry in `torch_entries()`, `native_entries()` or `compact_entries()`.
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
    group: str                 # "native", "compact", "neural" or "blend"
    needs_torch: bool
    description: str
    run: Callable = None       # run(data, config) -> (probabilities, n_inputs, note)
    config: dict = None


class Data:
    """Chronological split built lazily.
    Default: train 2010-2013, validate 2014.  holdout=True: train 2010-2024, score 2025-26."""

    def __init__(self, path, quick=False, holdout=False):
        self.path, self.quick, self.holdout = path, quick, holdout
        self._full = self._compact = None

    def bounds(self):
        """(last training year, first scored year, last scored year)"""
        return (2024, 2025, 9999) if self.holdout else (2013, 2014, 2014)

    def full(self):
        if self._full is None:
            features, truth = training_table(load_data(self.path))
            year = market_days(features.index).year
            train_until, valid_from, valid_to = self.bounds()
            train, valid = year <= train_until, (year >= valid_from) & (year <= valid_to)
            self._full = (features.loc[train], truth.loc[train],
                          features.loc[valid], truth.loc[valid])
        return self._full

    def compact(self):
        if self._compact is None:
            self._compact = pm.prepare(self.path, *self.bounds())
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
    """The price MLP: every non-duplicate input, signed-log scaled and standardised."""
    x_train, t_train, x_valid, t_valid = data.full()
    columns, a, b = pm.full_inputs(x_train, x_valid)
    y_train, y_valid = t_train["price"].to_numpy(), t_valid["price"].to_numpy()
    if data.holdout:     # never select an epoch on the holdout: use the epoch count found on 2014
        probabilities, _, _ = pm.fit_mlp(a, y_train, b, y_valid, dropout=config["dropout"],
                                         decay=config["decay"],
                                         fixed_epochs=config["holdout_epochs"])
        return probabilities, a.shape[1], f"{config['holdout_epochs']} epochs, fixed"
    probabilities, epoch, _ = pm.fit_mlp(a, y_train, b, y_valid, dropout=config["dropout"],
                                         epochs=10 if data.quick else config["epochs"],
                                         patience=config["patience"], decay=config["decay"])
    return probabilities, a.shape[1], f"best epoch {epoch}"


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
              "logistic regression, C=0.1, 45 hand-picked inputs", run_compact_logistic, {"C": 0.1}),
        Entry("compact_boosted", "compact_boosted", "compact", False,
              "LightGBM, 31 leaves, 300 trees, 45 hand-picked inputs", run_compact_boosted,
              {"trees": 300}),
    ]


def torch_entries():
    return [Entry("mlp", "mlp", "neural", True,
                  "MLP ~535-64-32-1, dropout 0.3, early stopping, all non-duplicate inputs",
                  run_mlp, dict(pm.MLP_SETTINGS))]


BLEND_WEIGHTS = (0.5, 0.5)       # LightGBM, MLP
BLENDS = {"blend": "mlp"}        # blend name -> the MLP it averages with


def registry(quick):
    blend = Entry("blend", "blend", "blend", False,
                  "50/50 average of the strongest native LightGBM and the MLP")
    return native_entries(quick) + compact_entries() + torch_entries() + [blend]


def select(entries, tokens):
    names = {e.name for e in entries}
    chosen = []
    for token in tokens:
        if token == "all":
            matched = entries
        elif token in ("native", "compact", "blend"):
            matched = [e for e in entries if e.group == token]
        else:
            matched = [e for e in entries if token in (e.name, e.family)]
        if not matched:
            raise SystemExit(f"Unknown model {token!r}. Available: {sorted(names)} "
                             "or groups all/native/compact/blend (see --list)")
        chosen += [e for e in matched if e not in chosen]
    return chosen


# ---- scoring -----------------------------------------------------------------------------
def score(name, family, inputs, probabilities, y, seconds, note="", threshold=None):
    threshold = pm.best_threshold(y, probabilities) if threshold is None else threshold
    predicted = probabilities > threshold
    return {"model": name, "family": family, "inputs": inputs, "threshold": threshold,
            "accuracy": accuracy_score(y, predicted) * 100,
            "balanced_accuracy": balanced_accuracy_score(y, predicted) * 100,
            "balanced_accuracy_at_0.5": balanced_accuracy_score(y, probabilities > 0.5) * 100,
            "fit_seconds": seconds, "note": note}


def ba_interval(y, predicted, days, n=1000, seed=0):
    """95% interval for balanced accuracy, resampling whole days (they are strongly correlated)."""
    frame = pd.DataFrame({"pos": y == 1, "neg": y == 0, "pos_hit": (y == 1) & predicted,
                          "neg_hit": (y == 0) & ~predicted, "day": days})
    counts = frame.groupby("day").sum().to_numpy(dtype=float)     # pos, neg, pos_hit, neg_hit
    draws = np.random.default_rng(seed).integers(0, len(counts), (n, len(counts)))
    sample = counts[draws]
    ba = 100 * 0.5 * (sample[..., 2].sum(1) / sample[..., 0].sum(1)
                      + sample[..., 3].sum(1) / sample[..., 1].sum(1))
    return tuple(np.percentile(ba, [2.5, 97.5]))


def baseline_rows(data):
    x_valid, y = data.full()[2], data.y_valid()
    days = market_days(x_valid.index)
    rows = []
    for name, column in (("baseline: yesterday", "price_margin_1"),
                         ("baseline: last week", "price_margin_7")):
        predicted = x_valid[column].to_numpy() > 0
        rows.append({"model": name, "family": "baseline", "inputs": 0, "threshold": np.nan,
                     "accuracy": accuracy_score(y, predicted) * 100,
                     "balanced_accuracy": balanced_accuracy_score(y, predicted) * 100,
                     "balanced_accuracy_at_0.5": np.nan, "fit_seconds": 0.0, "note": ""})
        if data.holdout:
            rows[-1]["ci_low"], rows[-1]["ci_high"] = ba_interval(y, predicted, days)
    return rows


def failed(seconds, note):
    return {"prob": None, "inputs": np.nan, "seconds": seconds, "note": note}


def run_entries(entries, data):
    """Returns {name: dict(prob, inputs, note, seconds)}; failures have prob=None."""
    outputs = {}
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry.name}: {entry.description}", flush=True)
        started = time.time()
        try:
            probabilities, inputs, note = entry.run(data, entry.config)
            outputs[entry.name] = {"prob": np.asarray(probabilities, float), "inputs": inputs,
                                   "note": note, "seconds": time.time() - started}
            print(f"    done in {time.time() - started:.0f}s", flush=True)
        except Exception as error:   # keep going: one failing model should not lose the rest
            outputs[entry.name] = failed(time.time() - started,
                                         f"FAILED: {type(error).__name__}: {error}")
            print(f"    {outputs[entry.name]['note']}", flush=True)
    return outputs


def run_torch_worker(names, args):
    """Run PyTorch models in a child process and load the probabilities it saved."""
    with tempfile.TemporaryDirectory() as directory:
        command = [sys.executable, os.path.abspath(__file__), "--worker", "--out-dir", directory,
                   "--data", args.data] + (["--quick"] if args.quick else []) + \
                  (["--holdout", "--confirm-holdout"] if args.holdout else []) + \
                  ["--models"] + names
        result = subprocess.run(command)
        meta_path, preds_path = (os.path.join(directory, "meta.json"),
                                 os.path.join(directory, "preds.npz"))
        if result.returncode != 0 or not os.path.exists(meta_path):
            return {n: failed(0.0, f"FAILED: worker exit code {result.returncode}")
                    for n in names}
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        saved = np.load(preds_path)
        return {n: dict(m, prob=saved[n] if n in saved.files else None)
                for n, m in meta.items()}


def add_blend(outputs, name="blend", mlp="mlp"):
    """50/50 average of the strongest native LightGBM and an MLP (probabilities)."""
    boosted = [n for n in outputs if n.startswith("boosted_")]
    best = boosted[-1] if boosted else None          # the last candidate is the largest one
    a, b = outputs.get(best), outputs.get(mlp)
    if a is None or b is None or a["prob"] is None or b["prob"] is None:
        outputs[name] = failed(0.0, f"FAILED: needs a native LightGBM model and {mlp}")
        return
    outputs[name] = {
        "prob": BLEND_WEIGHTS[0] * a["prob"] + BLEND_WEIGHTS[1] * b["prob"],
        "inputs": np.nan, "seconds": a["seconds"] + b["seconds"],
        "note": f"{BLEND_WEIGHTS[0]:.1f} {best} + {BLEND_WEIGHTS[1]:.1f} {mlp}"}


def build_rows(outputs, families, data):
    y = data.y_valid()
    fixed = 0.5 if data.holdout else None          # never tune the threshold on the holdout
    days = market_days(data.full()[2].index)
    rows = []
    for name, out in outputs.items():
        if out["prob"] is None:
            rows.append({"model": name, "family": families[name], "inputs": np.nan,
                         "threshold": np.nan, "accuracy": np.nan, "balanced_accuracy": np.nan,
                         "balanced_accuracy_at_0.5": np.nan, "fit_seconds": out["seconds"],
                         "note": out["note"]})
        else:
            row = score(name, families[name], out["inputs"], out["prob"], y,
                        out["seconds"], out["note"], threshold=fixed)
            if data.holdout:
                row["ci_low"], row["ci_high"] = ba_interval(y, out["prob"] > fixed, days)
            rows.append(row)
    return rows + baseline_rows(data)


def print_table(rows, holdout=False):
    models = sorted([r for r in rows if r["family"] != "baseline"],
                    key=lambda r: -np.nan_to_num(r["balanced_accuracy"], nan=-1))
    baselines = [r for r in rows if r["family"] == "baseline"]
    header = f"{'model':20s} {'inputs':>6s} {'thr':>5s} {'acc %':>7s} {'bal.acc %':>10s} " \
             f"{'@0.5':>7s} {'secs':>6s}" + ("  95% interval " if holdout else "") + "  note"
    print("\n" + header + "\n" + "-" * len(header))
    for r in models + baselines:
        fmt = lambda v, f: "" if pd.isna(v) else format(v, f)  # noqa: E731
        print(f"{r['model']:20s} {fmt(r['inputs'], '.0f'):>6s} {fmt(r['threshold'], '.2f'):>5s} "
              f"{fmt(r['accuracy'], '.2f'):>7s} {fmt(r['balanced_accuracy'], '.2f'):>10s} "
              f"{fmt(r['balanced_accuracy_at_0.5'], '.2f'):>7s} {r['fit_seconds']:6.0f}  "
              + (f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]  " if holdout and not pd.isna(
                  r.get("ci_low", np.nan)) else ("               " if holdout else ""))
              + f"{r['note']}")
    if holdout:
        print("\nHOLDOUT: fit on 2010-2024, scored on 2025-26 at a fixed 0.5 threshold with a fixed "
              "MLP epoch count;\nnothing was tuned on these days. Intervals resample whole days "
              "(95%). Do not choose or tune\nmodels by this table; report it as a final check. 'inputs' differs between "
              "native and compact models.")
        return
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
    parser.add_argument("--holdout", action="store_true",
                        help="one-off check: fit on 2010-2024, score 2025-26 (needs --confirm-holdout)")
    parser.add_argument("--confirm-holdout", action="store_true",
                        help="acknowledge that this looks at the 2025-26 holdout")
    parser.add_argument("--out-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.holdout and not args.confirm_holdout:
        raise SystemExit("--holdout looks at the 2025-26 holdout. Decide which models and settings to "
                         "evaluate first, then add --confirm-holdout.")

    if args.worker:   # child process: only the PyTorch models, no LightGBM import
        entries = [e for e in torch_entries() if e.name in args.models]
        outputs = run_entries(entries, Data(args.data, args.quick, args.holdout))
        meta = {n: {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                    for k, v in o.items() if k != "prob"} for n, o in outputs.items()}
        np.savez(os.path.join(args.out_dir, "preds.npz"),
                 **{n: o["prob"] for n, o in outputs.items() if o["prob"] is not None})
        with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        return

    entries = registry(args.quick)
    if args.list:
        print(f"{'name':18s} {'group':8s} {'process':8s} description")
        for e in entries:
            print(f"{e.name:18s} {e.group:8s} {'torch' if e.needs_torch else 'main':8s} "
                  f"{e.description}")
        print("\nGroups: all, native, compact, blend. A family name selects every model of that "
              "family.\nNative = team models with their own inputs; compact = reference models on "
              "45 hand-picked inputs;\nmlp = the price MLP (all inputs); blend = LightGBM + mlp.")
        return

    chosen = select(entries, args.models)
    names = {e.name for e in chosen}
    blends = {n: m for n, m in BLENDS.items() if n in names}
    if blends:               # a blend needs its two components, so run (and show) them too
        needed = {[e for e in entries if e.family == "boosted"][-1].name} | set(blends.values())
        if needed - names:
            print(f"Blends also run {', '.join(sorted(needed - names))} (their components).")
        chosen = [e for e in entries if e.name in names | needed]
    if args.holdout:
        print("*** HOLDOUT MODE: fit on 2010-2024, scoring 2025-26. Threshold fixed at 0.5, MLP "
              "epochs fixed. ***")
    print(f"Running {len(chosen)} model(s): {', '.join(e.name for e in chosen)}"
          f"{'  [quick]' if args.quick else ''}\n", flush=True)

    data = Data(args.data, args.quick, args.holdout)
    outputs = run_entries([e for e in chosen if not e.needs_torch and e.name not in BLENDS], data)
    torch_names = [e.name for e in chosen if e.needs_torch]
    if torch_names:
        print("\nStarting a separate process for PyTorch models ...", flush=True)
        outputs.update(run_torch_worker(torch_names, args))
    for blend_name, mlp_name in blends.items():
        add_blend(outputs, blend_name, mlp_name)

    rows = build_rows(outputs, {e.name: e.family for e in chosen}, data)
    print_table(rows, args.holdout)
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(args.output, "comparison.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print("saved", path)


if __name__ == "__main__":
    main()
