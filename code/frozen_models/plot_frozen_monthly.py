"""Monthly error of the Step 1 frozen models, January 2015 - October 2026 (2025-26 included), with 95% intervals computed
two ways (Appendix A, analysis_tools.confidence_intervals): (a) half-hours i.i.d., (b) block bootstrap over whole days.

Frozen models (code/forecast_package/train.py select_models): candidates fitted on 2010-2013, validated on 2014, then
refitted on 2010-2014 and never updated. Best candidate per family on 2014:
  demand: LightGBM 63 leaves (900 trees, lr 0.025; MAPE 2.114%)  and Ridge alpha = 1000 (MAPE 2.944%)
  price:  LightGBM 31 leaves (600 trees, lr 0.04;  BA 81.74%)    and Logistic C = 0.01 (BA 79.27%)
Demand forecasts: saved run_experiments output (results/protocol/run_experiments/half_hour_predictions.parquet, branch
price-c2; read from the checkout if present, else via git show). Price: the saved forecasts are labels at 2014-tuned
cut-offs, so both price models are refitted here exactly as in Step 1 (2010-2014) to get probabilities, cut at 0.5. Refits use the Step 1 code (240 features) from branch price-c2.
Run from the repo root: python code/frozen_models/plot_frozen_monthly.py"""
import io, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); ROOT = os.path.dirname(CODE)
# The Step 1 run used the original 240-feature pipeline, which is the code/forecast_package of branch price-c2 (its
# feature list equals results/protocol/run_experiments/run_manifest.json). It is extracted to a temporary folder.
STEP1 = tempfile.mkdtemp(prefix="step1_pkg_")
for f in ("model.py", "train.py", "random_forest.py"):
    open(os.path.join(STEP1, f), "w").write(subprocess.run(
        ["git", "-C", ROOT, "show", f"origin/price-c2:code/forecast_package/{f}"], capture_output=True, text=True, check=True).stdout)
sys.path[:0] = [STEP1, CODE]
from model import load_data, training_table, market_days                      # noqa: E402
from train import fit_model                                                   # noqa: E402
from analysis_tools import confidence_intervals                               # noqa: E402
OUT = os.path.join(ROOT, "results", "frozen_models_2015_2026")
REL = "results/protocol/run_experiments/half_hour_predictions.parquet"
DEMAND = {"LightGBM (63 leaves)": "frozen_boosted_3_demand", "Ridge (α = 1000)": "frozen_linear_1_demand"}
PRICE = {"LightGBM (31 leaves)": ("boosted_2", {"family": "boosted", "leaves": 31, "trees": 600, "learning_rate": 0.04}, 0.57),
         "Logistic (C = 0.01)": ("linear_0", {"family": "linear", "regularization": 0.01}, 0.58)}
os.makedirs(OUT, exist_ok=True)

path = os.path.join(ROOT, REL)
saved = pd.read_parquet(path if os.path.exists(path) else io.BytesIO(
    subprocess.run(["git", "-C", ROOT, "show", f"origin/price-c2:{REL}"], capture_output=True, check=True).stdout))
assert saved.index.min() == pd.Timestamp("2015-01-01 00:30") and saved.index.max() == pd.Timestamp("2026-11-01 00:00")

# price: refit the two frozen classifiers on 2010-2014 and predict probabilities (cached)
cache = os.path.join(OUT, "price_probabilities.parquet")
if not os.path.exists(cache):
    features, truth = training_table(load_data(os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")))
    assert features.shape[1] == 240, features.shape
    days = market_days(features.index)
    train = days <= pd.Timestamp("2014-12-31"); test = days >= pd.Timestamp("2015-01-01")
    probs = {}
    for name, (key, config, _) in PRICE.items():
        model = fit_model("price", config, features.loc[train], truth.loc[train])
        probs[key] = model["estimator"].predict_proba(features.loc[test, model["features"]])[:, 1]   # as in select_models
        print(f"refitted {name} on {int(train.sum())} rows (2010-2014)", flush=True)
    pd.DataFrame(probs, index=features.index[test]).to_parquet(cache)
prob = pd.read_parquet(cache)
assert prob.index.equals(saved.index)
for name, (key, _, tuned) in PRICE.items():          # the refit must reproduce the saved labels at the tuned cut-off
    agree = np.mean((prob[key].to_numpy() > tuned).astype(int) == saved[f"frozen_{key}_price"].to_numpy())
    print(f"{name}: refit labels at the Step 1 cut-off {tuned} agree with the saved run on {100 * agree:.3f}% of half-hours")

days = market_days(saved.index); month = days.to_period("M")
actual = saved["nsw_demand"].to_numpy(); up = saved["price_up"].to_numpy()
series = {**{n: ("demand", 100 * np.abs(saved[c].to_numpy() - actual) / actual) for n, c in DEMAND.items()},
          **{n: ("price", ((prob[k].to_numpy() > 0.5).astype(int) == up).astype(float)) for n, (k, _, _) in PRICE.items()}}
rows = []
for name, (task, values) in series.items():
    for m in month.unique():
        sel = np.asarray(month == m)
        v = values[sel].reshape(-1, 48)
        ci = confidence_intervals(v if task == "demand" else np.zeros_like(v), v if task == "price" else np.zeros_like(v), bootstrap=1000)
        key = "MAPE" if task == "demand" else "accuracy"
        rows.append({"task": task, "model": name, "month": m.to_timestamp(), "days": len(v),
                     "value": v.mean() * (1 if task == "demand" else 100),
                     "iid_low": ci[f"{key}_iid_low"], "iid_high": ci[f"{key}_iid_high"],
                     "boot_low": ci[f"{key}_day_boot_low"], "boot_high": ci[f"{key}_day_boot_high"]})
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "frozen_monthly_metrics.csv"), index=False)
T["iid_width"] = T.iid_high - T.iid_low; T["boot_width"] = T.boot_high - T.boot_low
print(T.groupby(["task", "model"])[["value", "iid_width", "boot_width"]].mean().round(3).to_string())
for task, name in (("demand", "MAPE"), ("price", "accuracy at 0.5")):
    for n in T[T.task == task].model.unique():
        s = T[(T.task == task) & (T.model == n)]; yr = s.month.dt.year
        print(f"{n} {name}: 2015-24 months mean {s[yr <= 2024].value.mean():.2f}, 2025-26 {s[yr >= 2025].value.mean():.2f}")

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="frozen_"))
import matplotlib; matplotlib.use("Agg")                                     # noqa: E402
import matplotlib.pyplot as plt                                              # noqa: E402
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#344054',
                     'text.color': '#182230', 'xtick.color': '#475467', 'ytick.color': '#475467',
                     'axes.edgecolor': '#D0D5DD', 'legend.frameon': False, 'savefig.facecolor': 'white'})
# per model: line colour, bootstrap-interval colour (light), i.i.d.-interval colour (dark)
STYLE = {"LightGBM (63 leaves)": ("#2a78d6", "#9fc5f0", "#0b3d91"), "Ridge (α = 1000)": ("#eb6834", "#f6bfa3", "#8a2e0b"),
         "LightGBM (31 leaves)": ("#2a78d6", "#9fc5f0", "#0b3d91"), "Logistic (C = 0.01)": ("#1baf7a", "#a3e2cb", "#0b5e40")}


def figure(task, title, ylabel, name):
    fig, ax = plt.subplots(figsize=(15, 5.4))
    models = list(T[T.task == task].model.unique())
    for i, n in enumerate(models):
        s = T[(T.task == task) & (T.model == n)]
        line, boot, iid = STYLE[n]
        x = s.month + pd.to_timedelta(15 + (i - 0.5) * 7, unit="D")        # mid-month, models offset by a week
        ax.errorbar(x, s.value, yerr=[s.value - s.boot_low, s.boot_high - s.value], fmt='none', ecolor=boot,
                    elinewidth=2.6, capsize=0, label=f'{n}: 95% CI, day-block bootstrap', zorder=2)
        ax.errorbar(x, s.value, yerr=[s.value - s.iid_low, s.iid_high - s.value], fmt='none', ecolor=iid,
                    elinewidth=0.9, capsize=1.8, capthick=0.9, label=f'{n}: 95% CI, half-hours i.i.d.', zorder=3)
        ax.plot(x, s.value, color=line, lw=1.4, marker='o', ms=2.5, label=f'{n}', zorder=4)
    ax.axvline(pd.Timestamp("2025-01-01"), color='#667085', lw=1, ls=':')
    ax.text(pd.Timestamp("2025-01-20"), 0.98, '2025–26', transform=ax.get_xaxis_transform(), va='top', fontsize=9, color='#475467')
    ax.set(title=title, ylabel=ylabel, xlabel='Month', xlim=(pd.Timestamp("2014-12-01"), pd.Timestamp("2026-12-01")))
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for n in models for l in (n, f'{n}: 95% CI, day-block bootstrap', f'{n}: 95% CI, half-hours i.i.d.')]
    ax.legend([handles[j] for j in order], [labels[j] for j in order], loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=9)
    ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    print("saved", os.path.join(OUT, name + ".png"))


figure("demand", "Frozen demand models (trained 2010–2014): monthly MAPE", "MAPE (%) · lower is better", "frozen_demand_monthly_mape")
figure("price", "Frozen price-direction models (trained 2010–2014): monthly accuracy, cut-off 0.5", "Accuracy (%) · higher is better",
       "frozen_price_monthly_accuracy")
