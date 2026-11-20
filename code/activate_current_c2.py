"""Stage and document C2 after its full-data build has succeeded."""
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
build=ROOT/'results/current_c2_build'
assert json.loads((build/'completion.json').read_text())['status']=='built'
source=ROOT/'code/forecast_package'; staged=ROOT/'code/submission_package'
for name in ['model.py','train.py','mlp_train.py','demand_feature_columns.json','c2_price.py','c2_recipe.py','c2_candidate.json']:
    shutil.copy2(source/name,staged/name)
for path in (source/'artifacts').iterdir():
    if path.is_file(): shutil.copy2(path,staged/'artifacts'/path.name)
selection=json.loads((ROOT/'results/final_config.json').read_text())
record=dict(status='current',branch='feature/aligned-demand-features',adoption_branch='feature/current-c2',
    previous_current_commit='f92a38e',demand_features=405,price_features=597,
    demand_weather_shape=False,price_weather_shape=True,price_probability_threshold=.5,
    price_model='C2: five LightGBM members, mean probabilities',price_seeds=[9417,9418,9419,9420,9421],
    demand_model='Frozen 50/50 LightGBM + ten-epoch MLP',demand_blend={'LightGBM':.5,'MLP':.5},
    config=selection,artifacts='code/forecast_package/artifacts',staged_submission='code/submission_package',
    build_results='results/current_c2_build',previous_current_backup='results/previous_current_f92a38e',
    evaluation_note='Adoption uses saved C2 research; no new holdout tuning. C2 reports use multiple protocols and seed counts; see source_reports.',
    compatibility_note='Built locally with Python 3.12.12. Exact pinned Python 3.11 rebuild not verified.')
(ROOT/'results/current_model.json').write_text(json.dumps(record,indent=2))
p=ROOT/'README.md'; old=p.read_text(); a=old.index('## Current validated model'); b=old.index('Everything you need',a)
p.write_text(old[:a]+'''## Current adopted model

Price is **C2**, averaging probabilities from **five LightGBM models** (seeds
9417–9421), using **597 features**: the existing 571 plus 26 Sydney temperature
and solar weather-shape inputs. Each member has 63 leaves, 1,200 trees, learning
rate 0.02, feature sampling 0.50 and minimum leaf samples 300. Predict class 1
when the mean probability is strictly greater than **0.50**.

Demand remains the validated **50/50 blend of LightGBM and a ten-epoch MLP**
(405 → 128 → 64 → 1), using the same 405 features and unchanged fitted demand
artifacts. The context gate is disabled. Inference requires no PyTorch import;
training runs the MLP in an isolated subprocess.

`code/forecast_package/` and `code/submission_package/` contain identical source
and fitted artifacts. Run `python code/forecast_package/train.py CSV` to rebuild
both tasks from the supplied dataset with the adopted defaults.

The authoritative record is `results/current_model.json`. Build timings, package
checks and source reports are under `results/current_c2_build/`. The previous
current, commit **f92a38e** (571-feature single price LightGBM at 0.57), is preserved
under `results/previous_current_f92a38e/`; older experiment results retain their
original meanings and were not relabelled as C2 results.

Saved C2 price comparisons (balanced accuracy):

| Protocol | Current-recipe control | C2 | Notes |
| --- | ---: | ---: | --- |
| Biennial, 2015–2022 | 79.173% | 79.337% | Three seeds each, cutoff 0.50 |
| Biennial, 2023–2024 | 83.121% | 83.325% | Three seeds each, cutoff 0.50 |
| Biennial, 2025–October 2026 | 85.320% | 85.451% | Three seeds each, reused period |
| Fixed training through 2024, 2025–October 2026 | 85.316% | 85.621% | Single-seed control vs five-seed C2, cutoff 0.50 |

These protocols differ; they are not a single untouched holdout comparison.
The unchanged demand blend previously scored **3.180% MAPE** on its biennial
2025–October 2026 evaluation. Full-data output checks verify package correctness,
not out-of-sample accuracy.

Artifacts are built locally with **Python 3.12.12**. A complete rebuild under the
exact prescribed Python 3.11 environment remains a separate compatibility check.

'''+old[b:])
