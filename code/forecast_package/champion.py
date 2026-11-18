"""Champion price-direction model (2026-11-19). See results/walkforward/REVIEW.md.

Opt-in: `python train.py CSV --selection champion_selection.json` trains it (the package default is unchanged and the
shipped artifacts are NOT this model). Best on the 2015-2024 walk-forward (80.715 BA); on the 2025-26 holdout it scored
85.38 vs 85.62 for Suhan's price-c2 recipe (-0.24 [-0.49, +0.02]); see REVIEW.md before choosing it.

    P(up) = 0.7 * mean(10 LightGBM members) + 0.3 * mean(2 neural nets);  predict "up" when P(up) > 0.5

    LightGBM A (5 seeds): LightGBM-63 package settings, inputs = 571 features + D4 demand features + 26 weather-shape
    LightGBM B (5 seeds): Suhan's recipe (63 leaves, 1200 trees, lr 0.02, colsample 0.5, min_child 300),
                          inputs = 571 features + A5 demand features + 26 weather-shape
    NN 1: numpy MLP 256-256 (5-seed ensemble), inputs = package NN inputs + D4 demand features
    NN 2: numpy MLP 64 (5-seed ensemble), package NN inputs only

Demand features are built from a day-ahead demand forecast f (the package's 0.5 LightGBM + 0.5 MLP blend). At inference
f is the package's own forecast for the target day. For TRAINING rows f must be out-of-sample: oof_folds() rebuilds it with
yearly folds for 2011-2015 and refits every second year afterwards (the schedule of the evaluated walk-forward), each fold
trained only on market days before the forecast year(s). The demand-feature models train from 2011 (the first year with an
out-of-sample forecast); NN 2 trains on all rows.
"""
import numpy as np
import pandas as pd

WEATHER_SHAPE = []
for _col in ("syd_temp", "syd_solar"):            # Suhan's recipe.weather_shape_names(), same order
    for _o in (2, 4):
        WEATHER_SHAPE += [f"{_col}_before_{_o}", f"{_col}_after_{_o}", f"{_col}_ramp_{_o}", f"{_col}_ramp_{_o}_change_1d"]
    WEATHER_SHAPE += [f"{_col}_daily_{_s}_change_1d" for _s in ("mean", "min", "max")]
    WEATHER_SHAPE += [f"{_col}_peak_slot", f"{_col}_peak_slot_change_1d"]

DEMAND_CONFIG = {"family": "boosted", "leaves": 63, "trees": 900, "learning_rate": 0.025}
SEEDS = (9417, 9418, 9419, 9420, 9421)
LGBM_A = {"family": "boosted", "leaves": 63, "trees": 900, "learning_rate": 0.025}
LGBM_B = {"family": "boosted", "leaves": 63, "trees": 1200, "learning_rate": 0.02,
          "colsample_bytree": 0.5, "min_child_samples": 300}
NN_BASE = {"family": "neural_net", "hidden": [256, 256], "dropout": 0.5, "learning_rate": 0.0003,
           "weight_decay": 0.0001, "batch_size": 1024, "epochs": 10, "seeds": [0, 1, 2, 3, 4],
           "log": "price", "winsorise": None, "bootstrap": False}
NN_D4 = dict(NN_BASE)
NN_64 = dict(NN_BASE, hidden=[64])
LGBM_WEIGHT = 0.7
FIRST_DEMAND_YEAR = 2011


def base_columns(features):
    return [c for c in features.columns if c not in WEATHER_SHAPE]


def demand_features(X, f, days, variant):
    """Demand-forecast features (identical to the walk-forward harness): 'A5' (Anubhav's five) or 'D4'.
    X: the 571 base features; f: demand forecast per row; days: market day per row. Returns X plus the new columns."""
    f = np.asarray(f, dtype=np.float64)
    new = {"dfc": f}
    if variant == "A5":
        new.update(dfc_minus_lag1=f - X["nsw_demand_lag_1"], dfc_minus_lag7=f - X["nsw_demand_lag_7"],
                   dfc_minus_prev_mean=f - X["nsw_demand_previous_mean"],
                   dfc_minus_prev_max=f - X["nsw_demand_previous_max"])
    elif variant == "D4":
        new.update(dfc_minus_anchor=f - X["demand_anchor"], dfc_minus_lag1=f - X["nsw_demand_lag_1"],
                   dfc_minus_lag7=f - X["nsw_demand_lag_7"], dfc_ratio_lag1=f / X["nsw_demand_lag_1"])
        g = pd.Series(f, index=X.index).groupby(np.asarray(days))
        mean, mx, mn, sd = (g.transform(s).to_numpy() for s in ("mean", "max", "min", "std"))
        prev = pd.Series(f, index=X.index).groupby(np.asarray(days)).shift(1)
        nxt = pd.Series(f, index=X.index).groupby(np.asarray(days)).shift(-1)
        new.update(dfc_day_mean=mean, dfc_day_max=mx, dfc_day_min=mn, dfc_day_std=sd,
                   dfc_day_mean_minus_prev_mean=mean - X["nsw_demand_previous_mean"],
                   dfc_minus_day_mean=f - mean, dfc_minus_day_max=f - mx,
                   dfc_rank_in_day=g.rank(pct=True).to_numpy(),
                   dfc_ramp_prev=(pd.Series(f, index=X.index) - prev).fillna(0).to_numpy(),
                   dfc_ramp_next=(nxt - pd.Series(f, index=X.index)).fillna(0).to_numpy())
        slot = X["slot"].to_numpy().astype(int)
        prof = pd.DataFrame({"day": np.asarray(days), "slot": slot, "f": f}).pivot(index="day", columns="slot", values="f")
        prof = prof.reindex(columns=range(48))
        dmap = pd.Series(np.arange(len(prof)), index=prof.index)
        rows = dmap.loc[np.asarray(days)].to_numpy()
        base = X["nsw_demand_previous_mean"].to_numpy()
        for s in range(48):
            new[f"dfc_profile_slot_{s}"] = prof[s].to_numpy()[rows] - base
    else:
        raise ValueError(variant)
    out = X.copy()
    for k, v in new.items():
        out[k] = np.asarray(v, dtype=np.float32)
    return out


def member_inputs(kind, features, f, days):
    """Inputs of one member, with the evaluated column order: base, then demand features, then weather shape."""
    X = features[base_columns(features)]
    if kind == "A":
        out = demand_features(X, f, days, "D4")
    elif kind == "B":
        out = demand_features(X, f, days, "A5")
    elif kind == "NN_D4":
        return demand_features(X, f, days, "D4")
    elif kind == "NN_64":
        return X
    else:
        raise ValueError(kind)
    for c in WEATHER_SHAPE:
        out[c] = features[c].to_numpy()
    return out


def oof_folds(days):
    """[(training_end_year, forecast_years)]: yearly folds to 2015, then refits every second year (even years)."""
    years = sorted(int(y) for y in pd.unique(pd.DatetimeIndex(days).year) if y >= FIRST_DEMAND_YEAR)
    groups = {}
    for y in years:
        end = y - 1 if y <= 2015 else (y if y % 2 == 0 else y - 1) - 1
        groups.setdefault(end, []).append(y)
    return sorted(groups.items())


def members():
    """(name, kind, config, weight) for the 12 members; weights sum to 1."""
    out = [(f"lgbm_A_{s}", "A", dict(LGBM_A, seed=s), LGBM_WEIGHT / 2 / len(SEEDS)) for s in SEEDS]
    out += [(f"lgbm_B_{s}", "B", dict(LGBM_B, seed=s), LGBM_WEIGHT / 2 / len(SEEDS)) for s in SEEDS]
    out += [("nn_D4", "NN_D4", NN_D4, (1 - LGBM_WEIGHT) / 2), ("nn_64", "NN_64", NN_64, (1 - LGBM_WEIGHT) / 2)]
    return out


def probability(bundle, features, f, days, price_probability):
    """Weighted ensemble probability for rows of `features` (with weather-shape columns) and demand forecast f."""
    cache, total = {}, np.zeros(len(features))
    for m in bundle["members"]:
        if m["kind"] not in cache:
            cache[m["kind"]] = member_inputs(m["kind"], features, f, days)
        total += m["weight"] * price_probability(m["model"], cache[m["kind"]])
    return total
