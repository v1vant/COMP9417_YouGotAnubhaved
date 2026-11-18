# Best price-direction configuration so far: review notes

**Status:** final champion on the 2015-2024 walk-forward (2026-11-19). On the 2025-26 holdout, run once, it scores 85.38 BA against 85.62 for Suhan's price-c2 recipe. That gap is not significant, but the team is leaning towards Suhan's method. See "Holdout test", "Why Suhan's model wins on the holdout" and "Demand-feature fixes" below.
The champion is available in `forecast_package` as an **opt-in** selection: `python train.py CSV --selection champion_selection.json`. The package default, the shipped artifacts and `submission_package` are unchanged.

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
4. **Deployment.** `forecast_package` implements the champion as an opt-in selection (`champion.py`, `champion_selection.json`):
   - It builds out-of-sample demand forecasts for every price training row inside `train.py`, predicts demand before price at inference, and adds the weather-shape inputs.
   - Fitted by the package on days up to 2013, all 12 members reproduce the walk-forward 2014 predictions exactly (max |difference| 0.00e+00).
   - Full-data artifacts were **not** built, and the full training time has not been measured against the 2-hour budget.
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

## Holdout test (2025-01-01 to 2026-10-31, run once on 2026-11-19)

**Protocol:**
- Every model, including the demand forecaster, was fitted on market days up to 2024-12-31 only.
- Each model then predicted the 669 holdout days (32,112 half-hours) at the fixed 0.5 threshold. Nothing was tuned.
- The demand forecast's error on the holdout was 3.13% MAPE (2024: 3.01%).

Script: `code/walkforward/holdout_test.py`, the exact copy that was run. Outputs: `results/walkforward/holdout/` (`score.txt`, `holdout_results.csv`, every model's probabilities in `holdout_predictions.npz`, specs in `holdout_models.json`).

| Model (trained on ≤ 2024) | BA 2025-26 | BA 2025 | BA 2026 Jan-Oct | Champion minus model [95% CI] |
|---|---|---|---|---|
| Suhan's price-c2 recipe (C2, 5 seeds, weather shape, no demand) | **85.621** | 86.899 | 84.112 | −0.237 [−0.49, +0.02] |
| LightGBM-63, no demand | 85.494 | 86.590 | 84.198 | −0.110 [−0.39, +0.18] |
| **Champion** | **85.384** | 86.746 | 83.780 | |
| LightGBM B alone (Suhan + A5 + weather shape) | 85.371 | 86.685 | 83.836 | +0.013 [−0.18, +0.20] |
| Package LightGBM-31 | 85.316 | 86.593 | 83.756 | +0.068 [−0.22, +0.36] |
| LightGBM A alone (LightGBM-63 + D4 + weather shape) | 85.265 | 86.578 | 83.734 | +0.119 [−0.08, +0.31] |
| Previous best | 85.171 | 86.462 | 83.688 | +0.212 [+0.04, +0.38] |
| NN 1 alone (NN + D4) | 84.657 | 86.007 | 83.179 | +0.727 [+0.37, +1.08] |
| NN 2 alone (NN 64) | 84.193 | 85.584 | 82.514 | +1.190 [+0.78, +1.59] |
| Yesterday baseline | 76.975 | | | |

At 0.57 every model scores lower (by 0.13 to 0.51 BA). That comparison is descriptive only; no threshold was chosen on the holdout.

Suhan reported 85.80 and 85.45. Those figures come from his own protocols: models refitted every 2 months, or biennially with 2026 predicted by a model trained through 2025. Both used 3-seed bags. With our protocol (one model trained ≤ 2024), his recipe scores 85.58 with 3 seeds and 85.62 with 5.

## Why Suhan's model wins on the holdout

These are diagnostics on existing holdout predictions; no model was chosen from them. Every number was recomputed by an independent check.

- **The gap is small and near the threshold.**
  - It is −0.24 BA [−0.49, +0.02].
  - The two models rank half-hours equally well: AUC 0.9321 for both, log-loss 0.3324 for both.
  - They disagree on 876 of 32,112 decisions, mostly near p = 0.5. The champion is right on 400 of those and wrong on 476.
- **It lies in the LightGBM half.**
  - The champion's LightGBM half alone trails Suhan's model by −0.24.
  - Adding the NN half changes the champion by +0.005.
- **The demand step accounts for the largest share.**
  - Part B (Suhan's recipe + A5, trained from 2011) trails Suhan by −0.25 [−0.50, −0.01]. Part A trails by −0.36 [−0.64, −0.07].
  - A Shapley split gives the demand step about half the gap. Part A and the NN take −0.07 and −0.05. No single share is significant.
  - The same reversal shows in package LightGBM-63 with and without A5: +0.32 on the walk-forward, −0.33 on the holdout.
- **About half of the Part B gap is seed averaging.**
  - Averaging Suhan's 5 seeds gains +0.10 over his single seeds on the holdout; Part B's average gains −0.02.
  - Matched single-seed gaps are only −0.06 to −0.17, none significant.
  - Suhan's recipe also gained more from seed averaging on the walk-forward, in 8 of 10 years.
- **The mechanism is extra false "up" calls on working days.** Working days are the only segment that survives multiple-comparison correction. Most of Part B's loss falls in the 10:00-16:00 solar hours, where up-labels have become rare: 13% on the holdout. Day-to-day demand changes became less informative about price direction, but that evidence is mixed.
- **The forecast is not unusually bad.** The demand forecast's holdout error (3.13%) sits on the 2015-24 trend. Part B's loss does not grow with the day's forecast error.
- **Starting training in 2011 is a confound.** Part B starts in 2011 because the demand forecast does not exist for 2010. A walk-forward-only control (Suhan's recipe from 2011, no demand) separates the two effects:
  - Over 2015-24: A5 +0.32 [+0.17, +0.45]; the 2011 start −0.04.
  - In 2024: A5 about 0; the 2011 start −0.13. Neither 2024 figure is significant.
  - The holdout split would need a new holdout fit, so it was not run.
- **The decline was already visible in 2024.** On the walk-forward, the champion's edge over Suhan's model was −0.08 [−0.48, +0.30] that year. The NNs' deficit to Suhan's model has widened every year:

  | Period | NN+D4 vs Suhan | NN64 vs Suhan |
  |---|---|---|
  | 2022 | −0.43 | −0.60 |
  | 2024 | −0.91 | −1.20 |
  | Holdout | −0.96 | −1.43 |

  0.7 Suhan + 0.3 NN gained +0.21 [+0.13, +0.29] over 2015-24 but +0.02 in 2024. On the holdout it is −0.14 [−0.31, +0.03] (diagnostic).
- **Suhan's recipe is not selection-free on 2025-26.** His weather-shape and wide/slow-tree choices came from screens scored on 2024-25. This cannot explain why Part B trails, since Part B uses the same recipe.
- **Replica fidelity.** Our copy of his recipe matches his C2 on `origin/price-c2` in parameters, features, weather-shape inputs and seeds. The only differences are 44 rows of the partial first day, 2010-01-29, and one floating-point label tie.

## Demand-feature fixes (walk-forward 2015-2024 only; rules declared before results)

**Where the demand gain came from.** In LightGBM-63, with and without A5 (both trained from 2011), the gain came almost entirely from the 10:00-16:00 solar hours. Each figure below is that hour group's additive share of the period's BA gain:

| Period | Midday share | All hours | Midday demand-forecast MAPE |
|---|---|---|---|
| 2015-17 | +0.20 | +0.24 | 2.4% |
| 2018-20 | +0.34 | +0.46 | 3.1% |
| 2021-23 | +0.22 | +0.36 | 5.2% |
| 2024 | −0.07 | −0.16 | 6.0% |

Over the same period:
- Midday demand fell from 8,416 MW (2015) to 6,698 MW (2024); evening demand stayed near 9,000 MW.
- Forecast error outside the solar hours stayed flat.

Scripts: `solar/solar_by_slot.py`, `solar/solar_contrib.py`.

**Pre-declared rules.**
- **Fallback:** F = 0.7 × Suhan (5 seeds) + 0.3 × NN 64.
- **When a fix works:**
  - (W1) it beats F over 2015-24 with the CI lower bound above 0;
  - (W2) its gain over F is at least 0 in 2024 and in 2022-24.
- **Extra gate for fix (b):** its demand forecast must cut midday MAPE over 2021-24 by at least 10% against a control rebuild, without raising all-hours MAPE.

**Fix (b): a solar-aware demand forecaster. FAILED its gate.**
- **Inputs added to Anubhav's recipe:**
  - Suhan's 26 weather-shape inputs;
  - six past-only solar-sensitivity inputs: the 56-day OLS slope of midday demand on irradiance, which went from −0.95 in 2015 to −4.77 in 2024; a 28-day midday/evening demand ratio; and the slope × irradiance changes.
- **Result:** midday MAPE over 2021-24 fell from 5.28% to 5.03%, which is −4.9% [−6.2, −3.5]. All-hours MAPE 2015-24 fell from 2.28% to 2.24%.
- The price models were not refitted.

Scripts: `solar/build_solar_features.py`, `solar/run_fixes.py`, `solar/mlp_fold.py`, `solar/gate_b.py`.

**Fix (a): demand members only outside 10:00-16:00. PASSED, narrowly.**
- In the solar hours, each demand member is replaced by a demand-free twin trained identically:
  - LightGBM-63 + weather shape from 2011;
  - Suhan's recipe from 2011;
  - NN 256 from 2011.
- NN 64 is unchanged.

Script: `solar/score_a.py`.

| Model | BA 2015-24 | 2015-19 | 2020-24 | 2022-24 | 2024 |
|---|---|---|---|---|---|
| Champion | **80.715** | 79.834 | 81.497 | 82.138 | 84.285 |
| Fix (a): gated champion | 80.523 | 79.612 | 81.343 | 82.075 | 84.351 |
| F = 0.7 Suhan + 0.3 NN 64 | 80.394 | 79.585 | 81.120 | 81.774 | 84.350 |
| Demand-free champion (twins at all hours; secondary) | 80.342 | 79.534 | 81.064 | 81.766 | 84.331 |
| Suhan alone | 80.232 | 79.394 | 80.994 | 81.767 | 84.367 |

- **Fix (a) minus F:** +0.13 [+0.03, +0.22] over 2015-24, +0.30 [+0.14, +0.47] over 2022-24, and **+0.001** [−0.23, +0.26] in 2024. Both rules pass, but W2 passes by 0.001 BA in 2024.
- **Fix (a) minus the champion:** −0.19 [−0.26, −0.11] over 2015-24, and +0.07 [−0.15, +0.29] in 2024.
- **Without any demand features,** the champion's extra parts (LightGBM A and the NNs) add nothing over F: −0.05 [−0.11, +0.01].

**Reading.**
- Almost all of the champion's walk-forward edge is the demand features.
- Gating them out at midday removes the 2024 loss but also most of the edge.
- In 2024 every candidate is within ±0.1 of Suhan's model.
- The gated champion was not run on the holdout; that would need new holdout fits of the twins.

**Data note.** The Sydney solar column peaks at slot 25 (12:30-13:00) in December-March and at slot 23 otherwise. The weather timestamps therefore appear to be local daylight-saving time, while the market data use NEM time. This shifts the weather-shape inputs and the solar-hours window by one hour in summer, for every model alike.

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
  It also holds Suhan's recipe (`suhan_c2ws_s*`), the demand-free twins (`suhan_c2ws_from2011_s*`, `lgbm63ws_from2011_s*`) and the fix (b) demand streams (`ctrl_blend_forecast.npy`, `solar_blend_forecast.npy`).
- `holdout_test.py`: the holdout test as run (provenance copy, not re-run).
- `solar/`: the solar-hours analysis and the two demand-feature fixes. Run them from `code/walkforward` (for example `python solar/score_a.py`); `build_solar_features.py` rebuilds the git-ignored `runs/solar_sens.npy`.
- `results/walkforward/holdout/`: holdout scores, predictions and model specs.
