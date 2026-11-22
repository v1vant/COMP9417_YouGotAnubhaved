# COMP9417: Day-ahead electricity forecasting

This project forecasts New South Wales electricity demand and price direction for the next 48 half-hour intervals. It uses the supplied National Electricity Market and weather dataset to compare models, study distribution shift, and evaluate retraining strategies chronologically. The price target is whether each next-day NSW spot price exceeds the **previous market day's mean price**. See [the project specification](specs/comp9417_project_spec.md) for the full protocol and marking rules.

The current model on `main` is:

- **Demand:** a 50/50 blend of one 63-leaf LightGBM and the mean of five ten-epoch MLPs.
- **Price direction:** five C2 LightGBMs; their mean probability is compared with a 0.50 cutoff.

The current configuration and links to its evidence are in [`results/current_model.json`](results/current_model.json). Historical experiments use other models and cutoffs; check each result's configuration before comparing scores. The 2025–October 2026 period has been inspected during development and is not an untouched test set.

## Repository guide

| Path | Purpose |
| --- | --- |
| `data/nem_weather_halfhourly_2010_2026.csv` | The fixed released dataset; see [data/README.md](data/README.md). |
| `harness/dayahead.py` | Authoritative market-day boundaries, targets, baselines and metrics. |
| `code/forecast_package/` | Current forecasting code, training scripts and fitted artifacts. |
| `code/submission_package/` | Staged copy of the current package, including its dependencies and artifacts. |
| `code/run_experiments.py`, `code/analysis_tools.py` | Original chronological model selection, drift detection and adaptation experiments. |
| `code/tests/` | Tests for features, targets, model inference and temporal leakage. |
| `results/` | Saved configurations, predictions, metrics and experiment provenance. |
| `starter_notebook.ipynb`, `example_package/`, `template/` | Course exploration notebook, baseline example and starter interface. |

In the package, `model.py` builds features and implements `Forecaster`; `train.py` fits the selected LightGBMs and starts MLP training; `mlp_train.py` exports the MLPs for NumPy inference. `c2_price.py` and `c2_recipe.py` build and fit the price ensemble. The two package directories intentionally mirror each other; preserve their source parity when updating the submission.

## Setup

Run the commands from the repository root. Use Python **3.11**, the version specified for marking:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows Command Prompt: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch 2.15.0 is required for rebuilding and the MLP training tests; the fitted package performs inference with NumPy, without importing PyTorch. Install PyTorch separately:

```bash
python -m pip install torch==2.15.0
```

On Linux or Windows without an NVIDIA GPU, the CPU build can be installed instead:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```

Use the exact versions in `requirements.txt` for marking compatibility. The checked-in current artifacts were built locally with Python 3.12.12; a full rebuild under the pinned Python 3.11 environment has not been verified. PyTorch 2.15 is not available for Intel-based Macs; use a supported environment such as a CSE lab machine or Colab for training.

## Run the code

### Run the tests

```bash
python -m unittest discover -s code/tests -v
```

These include synthetic checks for the day-ahead interface, baseline targets, ensemble inference and training-cutoff leakage. The MLP training test is skipped if PyTorch is unavailable. For only the cutoff checks:

```bash
python -m unittest discover -s code/tests -p test_training_cutoff_leakage.py -v
```

### Produce a forecast from the fitted package

The following example predicts the **last complete day in the supplied CSV**. It is an interface example, not an out-of-sample accuracy evaluation: those dates were used in training. Use the same idea with genuinely new history and next-day weather for deployment.

```python
# Run from the repository root, e.g. save this as forecast_example.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path('code/forecast_package').resolve()))
from model import Forecaster, WEATHER, load_data, market_days

frame = load_data('data/nem_weather_halfhourly_2010_2026.csv')
days = market_days(frame.index)
target_day = days.max()
history = frame.loc[days < target_day]
weather_next = frame.loc[days == target_day, WEATHER]

model = Forecaster('code/forecast_package/artifacts')
prediction = model.predict(history, weather_next)
print(prediction.head())  # nsw_demand_mw and nsw_price_up, 48 rows total
```

Each timestamp marks the **end** of a half-hour interval in market time (AEST, UTC+10, no daylight saving). Market day D contains D 00:30 through D 23:30 and D+1 00:00. A forecast may use market history through the end of D, calendar information and weather for D+1, but not D+1 demand or prices.

### Rebuild the current fitted models

```bash
python code/forecast_package/train.py data/nem_weather_halfhourly_2010_2026.csv
```

Without `--selection`, the training script uses its current default: one demand LightGBM, five price C2 LightGBMs and five demand MLPs. It trains on the supplied CSV and writes to **`code/forecast_package/artifacts/`**, replacing the fitted models there. This can take substantial time and requires PyTorch. The staged `code/submission_package/` is **not** automatically updated by this command; check and synchronise it deliberately after validating a rebuild. A `--quick` run uses reduced tree settings and is not the final model; it also writes to the artifacts directory.

### Reproduce the original historical experiments

```bash
python code/run_experiments.py --output results/local_run --resume
```

This runs the original Stage 1 selection and day-by-day frozen/adaptation simulation, writing metrics, plots and caches under `results/local_run/`. It is computationally expensive. `--quick` reduces the tree count but still runs the full chronological simulation, so its scores are not final-model results. The output path and code are for the **original model comparison**, not the complete five-MLP/C2 evaluation; consult the relevant saved result folders for those later experiments.

The older `python code/run_project.py --resume` pipeline runs tests, writes to the root `results/` directory, and retrains `code/forecast_package/artifacts/` using the simulation's `final_config.json`. That selection can differ from the current five-MLP/C2 configuration. Use the separate commands above when preserving the current artifacts and saved results matters.

### Explore and package

Open the introductory notebook with `python -m notebook starter_notebook.ipynb`. For submission, package the **contents** of `code/submission_package/`, including `model.py`, `train.py`, `mlp_train.py`, `c2_price.py`, `c2_recipe.py`, configuration files and `artifacts/`; the source imports the accompanying modules. For example, on macOS or Linux:

```bash
cd code/submission_package
zip -r ../../submission.zip . -x '*/__pycache__/*' '*.pyc'
```

Run the course-provided submission checker before submitting; it is not included in this cleaned repository. The marking environment has no network access, allows writes only to `/tmp`, and limits artifacts to 100 MB, memory to 4 GB and inference to 2 CPUs. See the specification for the full interface and time limits.

Market data: Australian Energy Market Operator (AEMO). Weather data: [Open-Meteo](https://open-meteo.com/), CC BY 4.0.
