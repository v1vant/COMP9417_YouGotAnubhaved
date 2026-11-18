# Price research v3: live-horizon recency and daily-count checks

v3 imports `code/price_v2` unchanged (its source hashes are recorded in earlier
status/frozen-selection files) and never modifies the demand recipe, existing
results or the staged submission. Run from the repository root in the pinned
`comp9417` environment:

```bash
python -m unittest discover -s code/price_v3 -p test_v3.py -v
python code/price_v3/diagnose.py                       # WP0, saved predictions only
python code/price_v3/live_horizon.py --variants V0 V0b V0c V1_hl2 V2_hl4 --workers 3   # WP1
python code/price_v3/level.py                          # WP2 selection, OOF, decoder screen, harness extras
python code/price_v3/live_horizon.py --variants L0_ctrl2011 L1_level --workers 3
python code/price_v3/live_horizon.py --schedule biennial --output results/price_v3_biennial \
    --variants V0 V1_hl2 V2_hl4 L0_ctrl2011 L1_level --workers 3
```

Each fit uses about 1.5-2 GB; three workers fit in 13 GB. The 571-feature table
is cached once as a memory map (`--cache`, default under the system temp dir,
about 0.7 GB) and is disposable. Completed fits are skipped on restart.

## What each step answers

* **WP0 `diagnose.py`** (`results/price_v3_diagnostics/`): per-year AUC,
  best-in-hindsight ("oracle") BA and realised BA for every saved candidate,
  separating ranking quality from cutoff choice; P0 error decomposition into the
  daily class-1 count versus the within-day ranking; errors by slot block, daily
  class fraction and distance from the threshold; monthly regime table; the
  weekly BA gain over the yesterday baseline (the live-mark quantity).
* **WP1 `live_horizon.py`** (`results/price_v3_live_horizon/`): 18 independent
  final-recipe checks. Each trains through a month end (Oct 2023-Aug 2026,
  every second month) and forecasts the next two months, tiling Nov 2023-Oct
  2026. This matches the live model's staleness (trained to 31 Oct, forecasting
  7-13 Dec); earlier evaluations used 12-24-month-old models. These are
  Stage-3 recipe checks, not a Stage-2 adaptation strategy, and the period is a
  reused, previously inspected one. All variants use the frozen 0.57 cutoff.
  Oracle BA is reported as a diagnostic only.
* **WP2 `level.py`** (`results/price_v3_level/`): a day-level LightGBM
  (cross-entropy on k/48) using only day-constant causal inputs. Configuration
  and decoder weight are chosen on 2014 after fitting through 2013;
  out-of-fold predictions use the price replacement boundaries. The decoder
  screen re-allocates P0's daily count without refitting P0; the stacked arm
  adds the out-of-fold prediction as two features with a matched 2011-start
  control. `harness_extra/` is regenerable and may be deleted.
* **`--schedule biennial`** reproduces P0's budget-compliant schedule (2015 fit,
  replacements 2016/18/20/22/24/26) for any variant: development 2015-2022,
  confirmation 2023-2024, reused 2025-Oct 2026.

`frozen_variants.json` records every variant before its fits are scored;
changing a registered configuration is refused. `gate.csv` applies the
pre-registered promotion rules from the plan.

## Round 2 (18 Nov): cutoff, weather shape, capacity

Planning probes (`screen.py`, saved to `results/price_v3_screen/`) found two levers:

* **Cutoff 0.50** (the natural cutoff for class-balanced training) instead of
  the 2014-tuned 0.57: +0.57 pp on 2015-2022 (biennial P0), about +0.15 pp for
  fresh models, and roughly zero in 2023-2026. It never hurt materially.
* **Weather-shape inputs** (26 Sydney temperature/solar shape columns already
  produced by `feature_frame(weather_shape=True)`; next-day weather only) and a
  **wider/slower LightGBM** improved AUC in single-cutoff screens on 2024-2025.

Pre-registered candidates (bags of seeds 9417/9418/9419, scored at 0.50 and 0.57):
C0 = P0; C1 = +weather shape; C2 = C1 + `num_leaves=63, learning_rate=.02,
n_estimators=1200, colsample_bytree=.5, min_child_samples=300`; C3 = C2 +
near-threshold down-weighting (scale 0.10). `recipe.py` is the single fitting
path shared with the package (tested equal to v2 `fit_model` for P0).

```bash
python code/price_v3/live_horizon.py --schedule biennial --output results/price_v3_biennial --variants V0b V0c --workers 2
python code/price_v3/live_horizon.py --variants C2_s9417 C2_s9418 C2_s9419 --workers 3       # likewise C1, C3
python code/price_v3/live_horizon.py --schedule biennial --output results/price_v3_biennial --variants C2_s9417 C2_s9418 C2_s9419 --workers 3
python code/price_v3/live_horizon.py --report-only --cutoffs 0.5 0.57 --variants V0 V0b V0c C1_s9417 ...
python code/price_v3/choose_candidate.py        # applies the frozen rule -> results/price_v3_final/frozen_selection_round2.json
python code/price_v3/export.py --candidate <package_candidate.json> --output /tmp/pkg --data data/nem_weather_halfhourly_2010_2026.csv
```

`choose_candidate.py` encodes the rule fixed before the runs: eligible only with a
positive pooled live-horizon change against C0_bag3 at 0.50, at least 10 of 18
windows improved, an AUC gain in both live sub-periods, and a non-negative BA plus
positive AUC change on biennial 2015-2022 (unseen by the screens). The best
eligible pooled BA wins, otherwise C0. The cutoff stays 0.50 unless 0.57 wins in
both live sub-periods.
