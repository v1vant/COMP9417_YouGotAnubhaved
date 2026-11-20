# COMP9417 26T3 Project Kit

## Current adopted model

**`main` is the canonical current branch.**

**Demand:** a frozen **50/50 blend** of one 900-tree LightGBM (seed 9417) and the
mean prediction of **five ten-epoch MLPs** (seeds 9417–9421). Each MLP is
405 → 128 → 64 → 1. All models use the existing 405 demand features. Each MLP
therefore has 10% of the final demand prediction; LightGBM has 50%. The context
gate is disabled. Inference is NumPy only; PyTorch is isolated to training.

**Price:** unchanged **C2**, averaging five LightGBMs (seeds 9417–9421) on
597 features: 571 base plus 26 Sydney temperature/solar shape features. Each
member has 63 leaves, 1,200 trees, learning rate 0.02, feature sampling 0.50,
minimum leaf samples 300, and the mean probability is compared strictly to 0.50.

`code/forecast_package/` and `code/submission_package/` have identical source
and fitted artifacts. Rebuild from the supplied data with
`python code/forecast_package/train.py CSV`; this trains one demand LightGBM,
five price LightGBMs, then all five demand MLPs in an isolated process.

The authoritative configuration is `results/current_model.json`. Build/export
checks and a clean rebuild smoke test are in `results/current_demand_mlp5_build/`.
The previous C2/single-MLP current (`30d9c60`) is preserved in
`results/previous_current_30d9c60/`; the older pre-C2 current (`f92a38e`) is
preserved in `results/previous_current_f92a38e/`. Earlier reports retain their
original model labels and should not be interpreted as the new current.

Demand comparisons (50/50 weights, chronological biennial schedule):

| Demand version | 2015–2024 MAPE | 2025–October 2026 MAPE | Holdout MAE |
| --- | ---: | ---: | ---: |
| Previous current: single LightGBM + single MLP | 2.2939% | 3.1797% | 219.78 MW |
| **Current: single LightGBM + five MLPs** | **2.2367%** | **3.1197%** | **215.46 MW** |

The holdout improvement is 0.0600 percentage points, paired seven-day bootstrap
95% CI [0.0361, 0.0834]; the period was inspected previously. It improved in
both years and 19/22 months. Full results, predictions and frozen protocol are
in `results/demand_single_lgbm_5mlp_holdout/`. Development results are in
`results/demand_seed_averaging_5seeds_2015_2024/`. Historical demand comparisons
are in `results/demand_holdout_model_comparison.csv`.

Saved C2 price comparisons (balanced accuracy):

| Protocol | Previous 31-leaf recipe | C2 | Notes |
| --- | ---: | ---: | --- |
| Biennial, 2015–2022 | 79.173% | 79.337% | Three seeds each, cutoff 0.50 |
| Biennial, 2023–2024 | 83.121% | 83.325% | Three seeds each, cutoff 0.50 |
| Biennial, 2025–October 2026 | 85.320% | 85.451% | Three seeds each, reused period |
| Fixed training through 2024, 2025–October 2026 | 85.316% | 85.621% | Single-seed control vs five-seed C2, cutoff 0.50 |

These price protocols differ from each other and must remain labelled.
The source reports are in `results/current_c2_build/source_reports/`.
Full-data package checks establish output validity and parity, not accuracy.

Artifacts are built locally with **Python 3.12.12**. A complete rebuild under the
exact prescribed Python 3.11 environment remains a separate compatibility check.

Everything you need for the project, apart from the specification on Moodle.

| Item | What it is |
| --- | --- |
| `data/nem_weather_halfhourly_2010_2026.csv` | The fixed dataset (see `data/README.md`) |
| `starter_notebook.ipynb` | Loads the data, builds a day-ahead table, computes both baselines |
| `template/model.py` | A blank `Forecaster` to start your model package |
| `example_package/` | A complete working package (a 7-day moving average): `model.py`, `train.py`, `artifacts/` |
| `code/` | Forecasting package, experiment code, training runner, model tests, and staged submission package |
| `harness/dayahead.py` | Shared day boundaries, targets, baselines, and metrics used by the notebook and model tests |
| `results/` | Saved experiments, selected training configuration, and validation results |
| `requirements.txt` | The exact library versions used in marking |

All commands below are run in a terminal (macOS/Linux) or in Command Prompt or PowerShell
(Windows), from inside the `comp9417_project_kit` folder.

## 1. Create a new environment (required)

Your models are saved with one set of library versions and loaded in marking with another
unless you match them exactly. Do **not** install into the Python you already use for other
courses: create a new environment with **Python 3.11** for this project.

**Option A: venv** (needs Python 3.11 installed, e.g. from https://www.python.org/downloads/)

macOS / Linux:
```
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows (Command Prompt):
```
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

Windows (PowerShell):
```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt.

**Option B: conda** (any operating system)
```
conda create -n comp9417 python=3.11
conda activate comp9417
```

Then, in the activated environment (on every system):
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**PyTorch (only if you use it).** Install version 2.15.0 in the same environment:
```
python -m pip install torch==2.15.0
```
On Windows or Linux without an NVIDIA GPU, the smaller CPU build is enough:
`python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.
PyTorch 2.15 is not available for Intel-based Macs; on those machines use a CSE lab
computer, Google Colab or another computer for PyTorch models.

Check the environment (it should print the versions in `requirements.txt`):
```
python -c "import sklearn, pandas, numpy; print(sklearn.__version__, pandas.__version__, numpy.__version__)"
```

Every time you open a new terminal, activate the environment again before working.

## 2. Days and timestamps

Each timestamp marks the **end** of a half-hour interval, in market time (AEST, UTC+10, no
daylight saving). Day D consists of the 48 intervals stamped D 00:30 to D 23:30 plus
(D+1) 00:00. When predicting day D+1, `history` ends with the interval stamped (D+1) 00:00.

## 3. Explore the data

```
python -m notebook starter_notebook.ipynb
```

## 4. Build your package

```
my_package/
├── model.py     defines class Forecaster
├── train.py     rebuilds artifacts/ from the released CSV only
└── artifacts/   your fitted models (at most 100 MB in total)
```

Start from `template/model.py`, or copy `example_package/`. The example is a 7-day moving
average: it shows the interface and how `artifacts/` is written and read, but it learns
nothing and is not a good forecaster. Rebuild its `artifacts/` with:
```
python example_package/train.py data/nem_weather_halfhourly_2010_2026.csv
```

Tips:
* Share one feature function between `train.py` and `model.py` (for example, import it in
  `train.py` from `model.py`), so that training and prediction compute features in the same way.
* `__init__` must only load files. `predict` receives the full history; use only what you
  need (the example uses the last 7 days) to stay well inside the 20-second limit.
* Build file paths with `os.path.join(artifact_dir, "...")`, never with `\` or `/` typed by
  hand, so that your code runs on any operating system.
* PyTorch: save `model.state_dict()` with `torch.save`, rebuild the network in `__init__` and
  load the weights with `torch.load(path, weights_only=True, map_location="cpu")`.

## 5. Run the project code

```
python code/run_project.py --resume
```

This runs the model tests, chronological experiments, and final training. Data and results
remain in the project root; fitted models are saved in `code/forecast_package/artifacts/`.
To run only the model tests:

```
python -m unittest discover -s code/tests -v
```

ZIP creation, ZIP checking, and clean-rebuild helper scripts have been removed.
Historical validation results have been preserved; the generated ZIP archives have been
removed. Rerunning training does not create archives.

In marking, packages run on Linux with no network access, where only `/tmp` is writable,
with 4 GB of memory and 2 CPUs. Keep your package within these limits.

## 6. Make the zip

The zip must contain `model.py`, `train.py` and `artifacts/`, either at the top level or
inside a single folder.

* macOS: right-click `my_package` in Finder and choose **Compress**.
* Windows: right-click `my_package` in File Explorer and choose
  **Send to > Compressed (zipped) folder** (on Windows 11: **Compress to ZIP file**).
* Linux or macOS terminal: `zip -r submission.zip my_package`

Rename the result to `submission.zip`. The original kit's submission checker is no longer
included in this cleaned project.

## Rules in brief

* Train only on the released CSV. No external data.
* No network access and no writing outside `/tmp` during prediction.
* The package should run on every day of the live week (one failed day costs 2 marks; more than one
  means no live-evaluation marks).

Weather data: Open-Meteo (https://open-meteo.com), CC BY 4.0. Market data: AEMO.
