"""Compare demand (regression) models under one protocol: fit 2010-2013, validate on 2014.

    python code/compare_demand_models.py --list                    # show what can be run
    python code/compare_demand_models.py --quick                   # everything, small models
    python code/compare_demand_models.py --models boosted mlp      # chosen types or exact names
    python code/compare_demand_models.py --models blend            # the deployed 50/50 blend
    python code/compare_demand_models.py --models all --output results/demand_comparison
    python code/compare_demand_models.py --holdout --confirm-holdout --models blend

Models (all use the team's own inputs from train.py / demand_mlp.py)
  linear_0, linear_1     Ridge regression, alpha 10 / 1000
  boosted_0, boosted_1   LightGBM regressor (L1 loss), 600 / 900 trees
  mlp                    demand MLP on the 405 columns in code/demand_feature_columns.json,
                         10 epochs, no early stopping (so nothing is tuned on 2014)
  blend                  0.5 * strongest LightGBM + 0.5 * MLP, as in the current Forecaster;
                         automatically runs the models it needs
Groups: all.  A family name (e.g. `linear`) selects every model of that family.

All models predict NSW demand in MW for every half-hour of 2014.  Metrics follow
harness/dayahead.py: MAPE (%) and MAE (MW).  "vs yesterday" is the relative MAPE improvement
over the same-half-hour-yesterday baseline, I = 100 * (1 - MAPE_model / MAPE_baseline), the
quantity the live-week marking scheme uses for the demand task.

These are 2014 validation scores, not test scores: 2014 was also used to choose the model
configurations in earlier work.

HOLDOUT MODE (--holdout --confirm-holdout): one-off check on 2025-26.  Models are fitted on every
year up to 2024 and scored on 2025-26, the closest analogue of the live week.  Nothing is tuned on
the holdout (the MLP already uses a fixed epoch count), and MAPE and the relative improvement
over yesterday come with 95% day-block bootstrap intervals.  Do not choose or tune models by
this result (that would turn the holdout into a selection set), and say in the report that the
holdout was used as a final check.  The saved package artifacts cannot be used for this: they
were trained through Oct 2026.

PyTorch and LightGBM must not share a process on macOS (their OpenMP runtimes deadlock), so the
MLP runs in a child process that hands its predictions back; the blend is computed here.

To add a model: write a `run_<name>(data, config)` returning (predictions_in_MW, n_inputs, note)
and add an Entry in `registry()` (or `torch_entries()` if it needs PyTorch).
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "harness"))
import dayahead as da  # noqa: E402  (authoritative MAPE / MAE)
from model import load_data, market_days, training_table  # noqa: E402

DEFAULT_DATA = os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")
MLP_COLUMNS = os.path.join(HERE, "demand_feature_columns.json")
BLEND_WEIGHTS = (0.5, 0.5)       # LightGBM, MLP: the frozen blend in the current Forecaster


@dataclass
class Entry:
    name: str
    family: str
    needs_torch: bool
    description: str
    run: Callable = None          # run(data, config) -> (predictions, n_inputs, note)
    config: dict = None


class Data:
    """Chronological split built lazily.
    Default: train 2010-2013, validate 2014.  holdout=True: train 2010-2024, score 2025-26."""

    def __init__(self, path, quick=False, holdout=False):
        self.path, self.quick, self.holdout, self._full = path, quick, holdout, None

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

    def y_valid(self):
        return self.full()[3]["demand"].to_numpy()


# ---- runners -----------------------------------------------------------------------------
def run_native(data, config):
    from train import fit_model           # imports LightGBM: never call this in the torch child
    from model import predict_task
    x_train, t_train, x_valid, _ = data.full()
    bundle = fit_model("demand", config, x_train, t_train)
    return predict_task(bundle, x_valid), len(bundle["features"]), ""


def run_mlp(data, config):
    from demand_mlp import fit_demand_mlp, predict_demand_mlp
    with open(MLP_COLUMNS, encoding="utf-8") as handle:
        columns = json.load(handle)
    x_train, t_train, x_valid, _ = data.full()
    epochs = 3 if data.quick else config["epochs"]
    bundle = fit_demand_mlp(x_train.loc[:, columns], t_train["demand"], epochs=epochs)
    return (predict_demand_mlp(bundle, x_valid.loc[:, columns]), len(columns),
            f"{epochs} epochs, fixed")


# ---- registry ----------------------------------------------------------------------------
def describe(config):
    if config["family"] == "linear":
        return f"Ridge regression, alpha={config['regularization']}"
    return f"LightGBM (L1 loss), {config['leaves']} leaves, {config['trees']} trees"


def native_entries(quick):
    from train import candidates          # imports LightGBM: skipped in the torch child
    entries, seen = [], {}
    for config in candidates("demand", quick):
        family = config["family"]
        index = seen[family] = seen.get(family, -1) + 1
        entries.append(Entry(f"{family}_{index}", family, False, describe(config),
                             run_native, config))
    return entries


def torch_entries():
    return [Entry("mlp", "mlp", True, "MLP 405-128-64-1, residual on the demand anchor, "
                  "weighted L1 loss, 10 epochs", run_mlp, {"epochs": 10})]


def registry(quick):
    blend = Entry("blend", "blend", False,
                  "50/50 average of the strongest LightGBM and the MLP (current Forecaster)")
    return native_entries(quick) + torch_entries() + [blend]


def select(entries, tokens):
    chosen = []
    for token in tokens:
        matched = entries if token == "all" else [
            e for e in entries if token in (e.name, e.family)]
        if not matched:
            raise SystemExit(f"Unknown model {token!r}. Available: "
                             f"{sorted(e.name for e in entries)} or 'all' (see --list)")
        chosen += [e for e in matched if e not in chosen]
    return chosen


# ---- execution ---------------------------------------------------------------------------
def run_entries(entries, data):
    """Returns {name: dict(pred, inputs, note, seconds)}; failures have pred=None."""
    outputs = {}
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry.name}: {entry.description}", flush=True)
        started = time.time()
        try:
            pred, inputs, note = entry.run(data, entry.config)
            outputs[entry.name] = {"pred": np.asarray(pred, float), "inputs": inputs,
                                   "note": note, "seconds": time.time() - started}
            print(f"    done in {time.time() - started:.0f}s", flush=True)
        except Exception as error:    # one failing model should not lose the others
            outputs[entry.name] = {"pred": None, "inputs": np.nan, "seconds": time.time() - started,
                                   "note": f"FAILED: {type(error).__name__}: {error}"}
            print(f"    {outputs[entry.name]['note']}", flush=True)
    return outputs


def run_torch_worker(names, args):
    """Run PyTorch models in a child process and load the predictions it saved."""
    with tempfile.TemporaryDirectory() as directory:
        command = [sys.executable, os.path.abspath(__file__), "--worker", "--out-dir", directory,
                   "--data", args.data] + (["--quick"] if args.quick else []) + \
                  (["--holdout", "--confirm-holdout"] if args.holdout else []) + \
                  ["--models"] + names
        result = subprocess.run(command)
        meta_path, preds_path = (os.path.join(directory, "meta.json"),
                                 os.path.join(directory, "preds.npz"))
        if result.returncode != 0 or not os.path.exists(meta_path):
            return {n: {"pred": None, "inputs": np.nan, "seconds": 0.0,
                        "note": f"FAILED: worker exit code {result.returncode}"} for n in names}
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        saved = np.load(preds_path)
        return {n: dict(m, pred=saved[n] if n in saved.files else None) for n, m in meta.items()}


def add_blend(outputs):
    boosted = [n for n in outputs if n.startswith("boosted_")]
    best = boosted[-1] if boosted else None      # the last candidate is the largest one
    a, b = outputs.get(best), outputs.get("mlp")
    if a is None or b is None or a["pred"] is None or b["pred"] is None:
        outputs["blend"] = {"pred": None, "inputs": np.nan, "seconds": 0.0,
                            "note": "FAILED: needs a LightGBM model and the MLP"}
        return
    outputs["blend"] = {
        "pred": BLEND_WEIGHTS[0] * a["pred"] + BLEND_WEIGHTS[1] * b["pred"],
        "inputs": np.nan, "seconds": a["seconds"] + b["seconds"],
        "note": f"{BLEND_WEIGHTS[0]:.1f} {best} + {BLEND_WEIGHTS[1]:.1f} mlp"}


# ---- scoring -----------------------------------------------------------------------------
def make_row(name, family, inputs, pred, y, seconds, note, baseline_mape=None):
    if pred is None:
        return {"model": name, "family": family, "inputs": np.nan, "MAPE": np.nan, "MAE": np.nan,
                "vs_yesterday_%": np.nan, "fit_seconds": seconds, "note": note}
    mape = da.mape(y, pred)
    return {"model": name, "family": family, "inputs": inputs, "MAPE": mape,
            "MAE": da.mae(y, pred),
            "vs_yesterday_%": np.nan if baseline_mape is None else 100 * (1 - mape / baseline_mape),
            "fit_seconds": seconds, "note": note}


def day_ape(y, pred, days):
    """Mean absolute percentage error of each day (every day has 48 half-hours)."""
    return pd.Series(100 * np.abs(y - pred) / y).groupby(days).mean().to_numpy()


def demand_intervals(model_days, base_days, n=1000, seed=0):
    """95% day-block bootstrap intervals for MAPE and for I = 100 * (1 - MAPE / MAPE_yesterday),
    resampling the same days for model and baseline."""
    draws = np.random.default_rng(seed).integers(0, len(model_days), (n, len(model_days)))
    mape, base = model_days[draws].mean(1), base_days[draws].mean(1)
    return (*np.percentile(mape, [2.5, 97.5]), *np.percentile(100 * (1 - mape / base), [2.5, 97.5]))


def build_rows(outputs, families, data):
    y, x_valid = data.y_valid(), data.full()[2]
    baselines = {"baseline: yesterday": x_valid["nsw_demand_lag_1"].to_numpy(),
                 "baseline: last week": x_valid["nsw_demand_lag_7"].to_numpy()}
    yesterday = da.mape(y, baselines["baseline: yesterday"])
    rows = [make_row(n, families[n], o["inputs"], o["pred"], y, o["seconds"], o["note"], yesterday)
            for n, o in outputs.items()]
    rows += [make_row(n, "baseline", 0, p, y, 0.0, "", yesterday) for n, p in baselines.items()]
    if data.holdout:
        days = np.asarray(market_days(x_valid.index))
        base_days = day_ape(y, baselines["baseline: yesterday"], days)
        predictions = {n: o["pred"] for n, o in outputs.items()} | baselines
        for row in rows:
            pred = predictions.get(row["model"])
            if pred is not None:
                (row["mape_low"], row["mape_high"], row["improvement_low"],
                 row["improvement_high"]) = demand_intervals(day_ape(y, pred, days), base_days)
    return rows


def print_table(rows, holdout=False):
    models = sorted([r for r in rows if r["family"] != "baseline"],
                    key=lambda r: np.nan_to_num(r["MAPE"], nan=1e9))
    baselines = [r for r in rows if r["family"] == "baseline"]
    header = (f"{'model':20s} {'inputs':>6s} {'MAPE %':>8s} {'MAE MW':>8s} "
              f"{'vs yest.':>9s} {'secs':>6s}"
              + ("  MAPE 95% interval   vs yest. 95% interval" if holdout else "") + "  note")
    print("\n" + header + "\n" + "-" * len(header))
    for r in models + baselines:
        fmt = lambda v, f: "" if pd.isna(v) else format(v, f)  # noqa: E731
        print(f"{r['model']:20s} {fmt(r['inputs'], '.0f'):>6s} {fmt(r['MAPE'], '.3f'):>8s} "
              f"{fmt(r['MAE'], '.1f'):>8s} {fmt(r['vs_yesterday_%'], '+.1f') + ('%' if not pd.isna(r['vs_yesterday_%']) else ''):>9s} "
              f"{r['fit_seconds']:6.0f}  "
              + (f"[{r['mape_low']:.2f}, {r['mape_high']:.2f}]   "
                 f"[{r['improvement_low']:+.1f}%, {r['improvement_high']:+.1f}%]   "
                 if holdout and not pd.isna(r.get("mape_low", np.nan)) else
                 ("                                          " if holdout else ""))
              + f"{r['note']}")
    if holdout:
        print("\nHOLDOUT: fit on 2010-2024, scored on 2025-26; nothing was tuned on these days. "
              "Intervals resample whole\ndays (95%). 'vs yest.' = 100 * (1 - MAPE / MAPE of "
              "yesterday); live-week marks use this\nquantity. Do not choose or tune models by "
              "this table; report it as a final check.")
        return
    print("\nSplit: fit 2010-2013, validate 2014 (validation, not test, scores). 'vs yest.' is the "
          "relative MAPE\nimprovement over the same-half-hour-yesterday baseline; the live-week "
          "marks use this quantity\n(5 marks need >= 40%).")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="model names, family names or 'all' (default: all)")
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

    if args.worker:       # child process: PyTorch models only, no LightGBM import
        entries = [e for e in torch_entries() if e.name in args.models]
        outputs = run_entries(entries, Data(args.data, args.quick, args.holdout))
        meta = {n: {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                    for k, v in o.items() if k != "pred"} for n, o in outputs.items()}
        np.savez(os.path.join(args.out_dir, "preds.npz"),
                 **{n: o["pred"] for n, o in outputs.items() if o["pred"] is not None})
        with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
        return

    entries = registry(args.quick)
    if args.list:
        print(f"{'name':12s} {'process':8s} description")
        for e in entries:
            print(f"{e.name:12s} {'torch' if e.needs_torch else 'main':8s} {e.description}")
        print("\nGroups: all. A family name selects every model of that family.")
        return

    chosen = select(entries, args.models)
    names = {e.name for e in chosen}
    if "blend" in names:    # the blend needs its two components, so run (and show) them too
        boosted = [e for e in entries if e.family == "boosted"]
        needed = {boosted[-1].name, "mlp"}
        if needed - names:
            print(f"'blend' also runs {', '.join(sorted(needed - names))} (its components).")
        chosen = [e for e in entries if e.name in names | needed]
    if args.holdout:
        print("*** HOLDOUT MODE: fit on 2010-2024, scoring 2025-26. Nothing is tuned on it. ***")
    print(f"Running {len(chosen)} model(s): {', '.join(e.name for e in chosen)}"
          f"{'  [quick]' if args.quick else ''}\n", flush=True)

    data = Data(args.data, args.quick, args.holdout)
    outputs = run_entries([e for e in chosen if not e.needs_torch and e.name != "blend"], data)
    torch_names = [e.name for e in chosen if e.needs_torch]
    if torch_names:
        print("\nStarting a separate process for PyTorch models ...", flush=True)
        outputs.update(run_torch_worker(torch_names, args))
    if "blend" in names:
        add_blend(outputs)

    families = {e.name: e.family for e in chosen}
    rows = build_rows(outputs, families, data)
    print_table(rows, args.holdout)
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(args.output, "comparison.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print("saved", path)


if __name__ == "__main__":
    main()
