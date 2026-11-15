"""Naive Bayes feature selection for price direction.

Default split: fit on 2010-2012, make every choice on 2013, test on 2014.

1. Correlation screen: Pearson correlation between every pair of features on the training
   years, computed on the values Naive Bayes sees (signed-log prices, temperature anomaly).
2. Fit Naive Bayes on every feature, score on the selection year.
3. Prune redundancy: rank features by stand-alone balanced accuracy on the selection year
   (--rank univariate, the default) or by evidence (--rank evidence), and keep a feature
   only if its |correlation| with every higher-ranked kept feature is below --max-corr.
4. Prune low impact: refit, then repeatedly drop the feature whose permutation costs the
   least selection-year balanced accuracy, until every feature is worth --min-drop points.

Features named with --keep (default: price_margin_7) skip both pruning steps, and do not
knock out their correlates in step 3.

The test year is reported for every stage but never used for any choice.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "forecast_package"))
from model import (NB_CATEGORIES, load_data, market_days, naive_bayes_continuous,
                   naive_bayes_terms, training_table)
from train import fit_model

THRESHOLDS = np.linspace(0.2, 0.8, 61)
# The earlier hand-picked set, kept as a reference row (model.NAIVE_BAYES now holds the
# features this script selected).
HAND_PICKED = ["weekday", "slot", "price_margin_1", "price_margin_7", "demand_anchor",
               "syd_cool", "syd_temp_anomaly", "annual_cos", "qld_price_loglag_1"]


def config_for(names):
    return {"family": "naive_bayes", "alpha": 1.0,
            "discrete": [name for name in names if name in NB_CATEGORIES],
            "continuous": [name for name in names if name not in NB_CATEGORIES]}


def probability(log_joint):
    log_joint = log_joint - log_joint.max(axis=1, keepdims=True)
    joint = np.exp(log_joint)
    return joint[:, 1] / joint.sum(axis=1)


class Scored:
    """A fitted NB model with its per-feature log-likelihood terms on a period."""

    def __init__(self, names, data, periods):
        self.model = fit_model("price", config_for(names), *data["train"])
        self.names = self.model["features"]               # discrete first, then continuous
        self.terms = {key: naive_bayes_terms(self.model["estimator"], data[key][0])
                      for key in periods}
        self.truth = {key: data[key][1]["price"].to_numpy() for key in periods}
        joint = self.log_joint("valid")
        scores = [balanced_accuracy_score(self.truth["valid"], probability(joint) > t)
                  for t in THRESHOLDS]
        self.threshold = float(THRESHOLDS[int(np.argmax(scores))])

    def log_joint(self, key):
        return self.model["estimator"]["log_prior"] + self.terms[key].sum(axis=1)

    def score(self, key, log_joint=None):
        joint = self.log_joint(key) if log_joint is None else log_joint
        return 100 * balanced_accuracy_score(self.truth[key], probability(joint) > self.threshold)

    def evidence(self):
        """Mean |log P(x_j|up) - log P(x_j|down)| on the selection year."""
        terms = self.terms["valid"]
        return pd.Series(np.abs(terms[:, :, 1] - terms[:, :, 0]).mean(axis=0), index=self.names)

    def univariate(self):
        """Balanced accuracy of a one-feature NB on the selection year, at its best threshold.
        NB fits each feature independently, so the one-feature model's posterior is a monotone
        function of that feature's log-likelihood ratio: no refitting needed."""
        terms, y = self.terms["valid"], self.truth["valid"]
        scores = {}
        for j, name in enumerate(self.names):
            fpr, tpr, _ = roc_curve(y, terms[:, j, 1] - terms[:, j, 0])
            scores[name] = float(100 * np.max((tpr + 1 - fpr) / 2))
        return pd.Series(scores)

    def permutation_drop(self, rng, repeats=3):
        """BA lost when one feature is shuffled. NB is additive in log space, so shuffling
        feature j is exactly shuffling its column of log-likelihood terms."""
        terms, joint = self.terms["valid"], self.log_joint("valid")
        base = self.score("valid", joint)
        drops = {}
        for j, name in enumerate(self.names):
            losses = []
            for _ in range(repeats):
                order = rng.permutation(len(terms))
                losses.append(base - self.score("valid", joint - terms[:, j] + terms[order, j]))
            drops[name] = float(np.mean(losses))
        return pd.Series(drops)

    def summary(self, label, years):
        return {"stage": label, "features": len(self.names), "threshold": self.threshold,
                f"BA_select_{years[0]}": round(self.score("valid"), 2),
                f"BA_test_{years[1]}": round(self.score("test"), 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-end", type=int, default=2012, help="last training year")
    parser.add_argument("--select-year", type=int, default=2013)
    parser.add_argument("--test-year", type=int, default=2014)
    parser.add_argument("--max-corr", type=float, default=0.4)
    parser.add_argument("--rank", choices=("univariate", "evidence"), default="univariate")
    parser.add_argument("--min-drop", type=float, default=0.1, help="BA points")
    parser.add_argument("--keep", nargs="*", default=["price_margin_7"],
                        help="features exempt from both pruning steps")
    parser.add_argument("--output", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()
    if not args.train_end < args.select_year < args.test_year:
        raise ValueError("Years must be chronological: train < select < test")
    keep = list(dict.fromkeys(args.keep))
    rng = np.random.default_rng(9417)

    features, truth = training_table(
        load_data(os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")))
    years = market_days(features.index).year
    data = {key: (features.loc[mask], truth.loc[mask]) for key, mask in
            (("train", years <= args.train_end), ("valid", years == args.select_year),
             ("test", years == args.test_year))}
    split = (args.select_year, args.test_year)
    all_names = list(features.columns) + ["syd_temp_anomaly"]
    unknown = [name for name in keep if name not in all_names]
    if unknown:
        raise ValueError(f"Unknown --keep features: {unknown}")
    stages = []

    # 1-2. Fit on everything; correlations on the values NB actually models.
    full = Scored(all_names, data, ("valid", "test"))
    stages.append(full.summary("all features", split))
    params, train_x = full.model["estimator"], data["train"][0]
    values = pd.concat([
        train_x[params["discrete"]].reset_index(drop=True).astype(np.float64),
        pd.DataFrame(naive_bayes_continuous(params, train_x), columns=params["continuous"]),
    ], axis=1)
    corr = values.corr().fillna(0.0)
    corr.round(4).to_csv(os.path.join(args.output, "nb_feature_correlation.csv"))
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    pairs = upper.abs().sort_values(ascending=False)
    print(f"{len(all_names)} features, {len(pairs)} pairs; "
          f"|r| >= {args.max_corr}: {int((pairs >= args.max_corr).sum())}, "
          f"|r| >= 0.95: {int((pairs >= 0.95).sum())}")

    # 3. Redundancy: the highest-ranked feature of each correlated group survives.
    # --keep features bypass the screen and do not knock out their correlates either.
    ranking = (full.univariate() if args.rank == "univariate" else full.evidence())
    ranking = ranking.sort_values(ascending=False)
    kept, removed_by = [], {}
    for name in ranking.index:
        if name in keep:
            continue
        partner = next((k for k in kept if abs(corr.at[name, k]) >= args.max_corr), None)
        if partner is None:
            kept.append(name)
        else:
            removed_by[name] = partner
    kept = keep + kept
    screened = Scored(kept, data, ("valid", "test"))
    stages.append(screened.summary(f"after correlation screen (|r| < {args.max_corr})", split))

    # 4. Impact: drop the least useful feature one at a time.
    current, dropped = screened, []
    while len(current.names) > len(keep) + 1:
        drops = current.permutation_drop(rng).drop(keep)
        weakest = drops.idxmin()
        if drops[weakest] >= args.min_drop:
            break
        dropped.append({"feature": weakest, "drop_points": round(float(drops[weakest]), 3)})
        current = Scored([n for n in current.names if n != weakest], data, ("valid", "test"))
    stages.append(current.summary(f"after impact pruning (>= {args.min_drop} pts each)", split))
    hand = Scored(HAND_PICKED, data, ("valid", "test"))
    stages.append(hand.summary("earlier hand-picked set (reference)", split))

    final = pd.DataFrame({"standalone_BA": ranking.reindex(current.names),
                          "evidence": current.evidence(),
                          "drop_points": current.permutation_drop(rng)})
    final["evidence_share_%"] = 100 * final["evidence"] / final["evidence"].sum()
    final["zero_inflated"] = pd.Series(
        current.model["estimator"]["zero_inflated"],
        index=current.model["estimator"]["continuous"]).reindex(final.index).fillna(False)
    final = final.sort_values("drop_points", ascending=False)
    result = {
        "settings": vars(args) | {"output": None},
        "stages": stages,
        "selected": config_for(list(final.index)),
        "final_features": final.round(4).reset_index(names="feature").to_dict("records"),
        "removed_as_correlated": removed_by,
        "removed_as_low_impact": dropped,
        "top_correlated_pairs": [{"a": a, "b": b, "r": round(float(corr.at[a, b]), 3)}
                                 for a, b in pairs.index[:30]],
    }
    with open(os.path.join(args.output, "nb_feature_selection.json"), "w", encoding="utf-8") as h:
        json.dump(result, h, indent=2)

    print("\nTop correlated pairs:")
    for row in result["top_correlated_pairs"][:12]:
        print(f"  {row['r']:+.3f}  {row['a']}  ~  {row['b']}")
    print(f"\nCorrelation screen kept {len(kept)} of {len(all_names)}; "
          f"impact pruning removed {len(dropped)} more.")
    print("\n" + pd.DataFrame(stages).to_string(index=False))
    print("\nSelected features:\n" + final[["standalone_BA", "evidence_share_%", "drop_points",
                                            "zero_inflated"]].round(3).to_string())
    if keep:
        print(f"(kept manually, never pruned: {', '.join(keep)})")


if __name__ == "__main__":
    main()
