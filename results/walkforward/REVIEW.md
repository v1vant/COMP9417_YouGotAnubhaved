# Best price-direction configuration so far: review notes

**Status:** final champion (2026-11-19). All remaining tests ran overnight. None met the champion rule, so the champion did not change; see "Final assessment" below.
This is a research result. It is **not** wired into `forecast_package` or `submission_package`.

## The champion

```
P(up) = 0.7 * mean of 10 LightGBM members  +  0.3 * mean of 2 neural nets,     predict "up" when P(up) > 0.5
```

| Part | Members | Settings | Inputs |
|---|---|---|---|
| LightGBM A | 5 seeds (9417–9421) | LightGBM-63 with our package settings: 63 leaves, 900 trees, lr 0.025, min_child 160, colsample 0.85, class-balanced | 571 package features + **D4** demand features + **26 weather-shape** inputs |
| LightGBM B | 5 seeds (9417–9421) | Suhan's recipe: 63 leaves, 1200 trees, lr 0.02, colsample 0.5, min_child 300, class-balanced | 571 features + **26 weather-shape** inputs + Anubhav's **A5** demand features |
| NN 1 | 5-seed ensemble | Our numpy MLP: 256-256, dropout 0.5, AdamW lr 3e-4 and weight decay 1e-4, 10 epochs | Package NN inputs + **D4** demand features |
| NN 2 | 5-seed ensemble | Our numpy MLP: 64 units, otherwise as NN 1 | Package NN inputs (no demand features) |

**Demand forecast.** All demand features use Anubhav's day-ahead demand forecast: 0.5 LightGBM + 0.5 ten-epoch MLP, rebuilt by `build_anubhav_blend.py`.
- 2011–2014 use his yearly out-of-sample folds (commit `4f32442`).
- 2015–2024 use his two streams, refitted on the team schedule.
- No demand value is fresher than the price model it feeds.

The demand-feature models and both LightGBM parts train from 2011, since the first out-of-sample forecasts are for 2011. NN 2 trains from 2010.

**Feature sets.**
- **A5:** forecast `f`, `f − lag_1`, `f − lag_7`, `f − previous_mean`, `f − previous_max`.
- **D4:** the D2 features; tomorrow's forecast day shape (mean/max/min/std, mean vs today's mean, slot minus day mean or max, within-day rank, ramps); and the 48-slot forecast profile minus today's mean demand.
- **Weather shape** (Suhan): 26 Sydney temperature and solar shape inputs from the supplied next-day weather. They cover ramps at ±2/±4 slots, peak slot, and day-over-day changes.

## Protocol

- **Walk-forward:** the team schedule, fitting up to 2013 to predict 2014, then ≤2014 → 2015, ≤2015 → 2016-17, ≤2017 → 2018-19, ≤2019 → 2020-21, ≤2021 → 2022-23 and ≤2023 → 2024. Every prediction is out-of-sample.
- **Threshold:** fixed at 0.5. The 70/30 weight is a priori, and members are equal-weighted inside each part.
- **Data:** only rows up to market day 2024-12-31 are read. The 2025-2026 holdout was not read, trained on or scored.
- **Confidence intervals:** paired day-block bootstrap, pooled over 2015-2024.
- **Champion rule** (declared before the final candidates were scored): replace the best so far if 2015-24 BA is higher, accuracy is not lower, and BA is not lower in either half. Among qualifying candidates the highest BA wins.

## Results, 2015-2024, threshold 0.5

| Configuration | BA 2015-24 | BA 2015-19 | BA 2020-24 | Accuracy 2015-24 | BA 2014 (tuning year) |
|---|---|---|---|---|---|
| **Champion** | **80.715** | **79.834** | **81.497** | **81.252** | 83.567 |
| Previous best (0.7 LGBM-63+A5 + 0.3 NN+D4, commit `feb079b`) | 80.572 | 79.750 | 81.300 | 81.112 | 83.366 |
| LightGBM A alone (5 seeds) | 80.529 | 79.624 | 81.336 | 81.130 | 83.341 |
| LightGBM B alone (5 seeds) | 80.462 | 79.631 | 81.205 | 81.118 | 83.501 |
| NN 1 alone | 80.021 | 79.239 | 80.678 | 80.316 | 82.630 |
| LightGBM-63, no demand | 80.006 | 79.154 | 80.776 | 80.599 | 83.142 |
| Package LightGBM-31 at 0.5 | 79.886 | 78.995 | 80.693 | 80.415 | 82.940 |
| NN 2 alone | 79.845 | 79.137 | 80.456 | 80.051 | 82.567 |
| Package LightGBM-31 at 0.57 (the reported 83.163% setup) | 79.448 | 78.311 | 80.503 | 80.509 | 83.124 |

### Champion minus each alternative

| Alternative | BA [95% CI] | Accuracy [95% CI] | Years better (of 10) |
|---|---|---|---|
| Previous best | **+0.14** [+0.06, +0.24] | +0.14 [+0.06, +0.23] | 8 |
| LightGBM A alone | +0.19 [+0.09, +0.28] | +0.12 [+0.03, +0.21] | 9 |
| LightGBM B alone | +0.25 [+0.16, +0.34] | +0.13 [+0.05, +0.22] | 10 |
| Package LightGBM-31 at 0.5 | **+0.83** [+0.67, +0.99] | +0.84 [+0.69, +0.98] | 10 |
| Package LightGBM-31 at 0.57 | **+1.27** [+1.09, +1.45] | +0.74 [+0.57, +0.92] | 10 |

### Where the gain comes from

All figures are on 2015-2024. Each gain is measured against the same model without the change.

| Ingredient | BA gain [95% CI] |
|---|---|
| Anubhav's demand features in LightGBM-63 | +0.31 [+0.14, +0.48] |
| D4 demand features in the NN | +0.42 [+0.28, +0.56] |
| Anubhav's forecast vs our LightGBM-only demand forecast | about +0.1 |
| Seed averaging, 1 → 3 seeds (LightGBM-63+D4+ws) | +0.07 [−0.00, +0.13] |
| Seed averaging, 3 → 5 seeds | +0.02 |
| Weather shape on Suhan's settings | +0.18 [+0.10, +0.27] |
| Weather shape on LightGBM-63+D4 (D4 already carries most of it) | +0.01 |
| Weather shape in the NN | +0.13 [+0.04, +0.21] |
| Blending LightGBM with the NN | +0.2 to +0.3 |
| Two LightGBM recipes (A + B) and two NNs (one with demand, one without), vs the best single pairing | about +0.04 to +0.07 |

## Caveats

1. **Selection on 2015-2024.**
   - The feature sets, the forecast source and the blend partners were compared on 2015-2024. Only the 0.5 threshold and the 70/30 weight are fixed a priori, and 2014 has almost no spike or negative-price days.
   - The champion is the less selective "broad" option, an equal-weight ensemble of every candidate component. The version with all 15 components scores 80.712, statistically identical.
   - Treat margins under about 0.15 BA between near-equal candidates as noise.
2. **Seed noise.** A single LightGBM varies by about 0.08 BA (sd) across seeds. The bootstrap intervals cover day-to-day sampling only.
3. **Weather.** Demand forecasts and weather-shape inputs use the supplied next-day weather, as every package feature does.
4. **Not deployable as is.** The package would need to:
   - build out-of-sample demand forecasts for every price training row (yearly folds 2011-2014, then the refit schedule) inside `train.py`;
   - predict demand before price at inference;
   - add the weather-shape inputs;
   - fit 10 LightGBM members and 2 NN ensembles within the 2-hour training budget.

   Retraining the deployed model on data through 2026-10 is the team's decision; it was not done here.
5. **Tested and not adopted (none beat its reference or the best so far):**
   - layered day-count / intraday-rank models and stacks;
   - continuous price-margin regression;
   - LightGBM or demand forecasts inside the NN as neurons;
   - kNN;
   - bigger NNs at the old settings;
   - recency weighting, top-k features and spike-specific features;
   - finer LightGBM bins.

   Negative-price features help the NN on the days after six or more negative slots (+1.2 BA with inputs capped at ±3 sd). The gain is neutral overall and in the blend.

## Final assessment of the remaining tests (overnight, 2026-11-19)

Each line is the paired BA gain on 2015-2024 [95% CI] against the matched reference. An idea counts as promising only if its CI is above 0 and it is not worse in either half. None qualified.

| Idea | Result vs reference | Promising? |
|---|---|---|
| Regression on the price margin, stacked into the classifier | −0.23 [−0.40, −0.06] | no |
| Focal loss γ=1 / γ=2 (vs the same custom path at γ=0) | +0.01 / −0.01 | no |
| Up-weighting rows past models were unsure about / near-threshold rows | −0.04 / −0.16 | no |
| NN with LightGBM as an output neuron, weight properly fitted | −0.49 vs the plain NN | no |
| Yamac's MLP (fixed 6 epochs / in-window early stopping) | −0.54 / +0.04 vs NN 256x256 | no |
| Anubhav's MLP | −2.8 vs NN 256x256 | no |
| Deeper NNs with the same 512 neurons (8x64 / 4x128 / 3x171) | +0.08 [−0.02, +0.18] / +0.04 / +0.02 | no |
| Wider NNs with other settings (512 or 1024 wide; lower lr, more decay, dropout 0.6, larger batch) | −0.09 to −0.30 | no |
| In-window (rolling) choice of NN size | +0.01 | no |
| Negative-price features in LightGBM-63+D4 (with or without weather shape) | −0.05 / −0.03 | no |
| Negative-price features in the NN, inputs capped at ±3 sd | +0.03 overall; +1.2 BA on the days after ≥6 negative slots | no (overall), subgroup fix |

Final assembly: the near-promising NNs (deep 8x64, deep 4x128, Yamac's early-stopped MLP) were also offered as extra ensemble members. No candidate met the champion rule. The closest were:
- 70/30 [LightGBM A] + [Yamac early-stopped MLP]: 80.755, but −0.02 on 2020-24.
- All 17 components: 80.717, but −0.01 on 2020-24.

Both are within noise of the champion (+0.04 [−0.03, +0.11] and +0.002).

**Notes for future work:**
- Price spikes are already captured. Spike days are the easiest days. The NN's signed log is what protects them, and finer LightGBM bins or spikiness features do not help.
- The NN's weakness after negative-price days comes from extreme out-of-range inputs in a regime that was absent before 2020. Capping standardised inputs fixes the blow-up.

## Reproduce

From the repo root with the project venv:

```
python code/walkforward/reproduce_best.py
```

This rebuilds the data cache from the repo CSV (rows up to 2024-12-31 only) and scores the shipped predictions.

To refit a member and compare it with the shipped predictions:

```
python code/walkforward/reproduce_best.py --refit lgbm63_D4ws_s9417 suhan_c2ws_A5_s9417 nn64 --fit-ends 2013
```

The targets, weather-shape and negative-price inputs are rebuilt locally. In the run above, all three refits matched the shipped predictions exactly (max |difference| 0.00e+00).

## Files

- `code/walkforward/harness.py`: data cache, team schedule, demand / weather-shape / negative-price feature builders, and one walk-forward fit (`fit_job`, kinds `sk`, `lgbx`, `nnx`, ...).
- `lgbx.py`: LightGBM variants, including seeds and parameter overrides.
- `nn_ext.py`: the extended numpy NN. It is identical to the package NN when no option is set.
- `layered.py`: the layered experiments (not in the champion).
- `build_anubhav_blend.py`, `build_targets.py`, `build_weather_shape.py`, `build_negative_features.py`: rebuild the inputs. `build_anubhav_blend.py` needs commit `4f32442`.
- `evaluate.py`: labels and the prediction loader.
- `best_so_far.py`: the paired comparison tables.
- `reproduce_best.py`: the entry point above.
- `results/walkforward/runs/`: out-of-sample probabilities for every fit of the champion's components and the references (`<config>__<fit_end>.npz`, keys `y<year>`), plus `configs.json` (exact specs) and `anubhav_blend_forecast.npy`.
