"""Frozen Step 1 models (trained on 2010-2014, never retrained): daily error with the days the two online detectors signal,
January 2015 - October 2026. No retraining is simulated; detectors use the fixed 2014 baseline.

Input detector: the weather drift score of branch `experiments` (plots/weather_drift.py, Suhan Karaca, commit 7b3be21,
extracted at run time): Mahalanobis distance of each day's seasonally adjusted weather summaries, climatology and covariance
fitted on 2010-2013; reference 2014.
Error detector: the frozen model's daily error (MAPE % / % of half-hours wrong); reference: the daily 2014 errors of the same
model fitted on 2010-2013 (saved by simulate_drift.py).
Both: analysis_tools.OnlineDetector (team code): 28-day rolling mean; threshold = 99th percentile of the 2014 rolling means;
at least 28 days between signals. Day D is observed at the end of D.
Run from the repo root after simulate_drift.py: python code/drift_detectors/plot_frozen_triggers.py"""
import json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); ROOT = os.path.dirname(CODE)
BASE = os.path.join(ROOT, "results", "drift_detectors_2015_2026")
OUT = os.path.join(BASE, "frozen_triggers")
CSV = os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv")
NAMES = {"demand_lightgbm63": "Demand · LightGBM (63 leaves)", "demand_ridge1000": "Demand · Ridge (α = 1000)",
         "price_lightgbm31": "Price · LightGBM (31 leaves), cut-off 0.5", "price_logistic001": "Price · Logistic (C = 0.01), cut-off 0.5"}
LINE = {"demand_lightgbm63": "#2a78d6", "demand_ridge1000": "#eb6834", "price_lightgbm31": "#2a78d6", "price_logistic001": "#1baf7a"}
SIG = {"input": "#7F56D9", "error": "#B42318"}
IDX = pd.date_range("2015-01-01", "2026-10-31")
os.makedirs(OUT, exist_ok=True)

tmp = tempfile.mkdtemp(prefix="weather_drift_")
open(os.path.join(tmp, "weather_drift.py"), "w").write(subprocess.run(
    ["git", "-C", ROOT, "show", "origin/experiments:plots/weather_drift.py"], capture_output=True, text=True, check=True).stdout)
sys.path[:0] = [tmp, CODE]
from weather_drift import daily_weather, weather_scorer       # noqa: E402
from analysis_tools import OnlineDetector                    # noqa: E402

w = daily_weather(CSV)
wscore = weather_scorer(w[w.index.year <= 2013])(w)                # climatology + covariance fitted on 2010-2013
inp = OnlineDetector(wscore[wscore.index.year == 2014].to_numpy())
inp_sig = [d for d in IDX if inp.observe(d, float(wscore.loc[d]))]
print(f"input (weather) detector: threshold {inp.threshold:.4f}; {len(inp_sig)} signals; first {[str(d.date()) for d in inp_sig[:3]]}")

rows, sigs = [], {}
for k in NAMES:
    r = json.load(open(os.path.join(BASE, f"{k}.json")))
    err = pd.Series(r["frozen_daily_error"], index=IDX)
    det = OnlineDetector(r["error_reference_daily"])
    err_sig = [d for d in IDX if det.observe(d, float(err.loc[d]))]
    sigs[k] = (err, err_sig, det.threshold)
    for p, m in (("2015-24", IDX.year <= 2024), ("2025-26", IDX.year >= 2025)):
        rows.append({"model": NAMES[k], "period": p, "frozen mean daily error": round(err[m].mean(), 3),
                     "error signals": sum(1 for d in err_sig if (d.year <= 2024) == (p == "2015-24")),
                     "input (weather) signals": sum(1 for d in inp_sig if (d.year <= 2024) == (p == "2015-24"))})
    print(f"{NAMES[k]}: error threshold {det.threshold:.4f}; {len(err_sig)} error signals; first {[str(d.date()) for d in err_sig[:3]]}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "frozen_trigger_counts.csv"), index=False)
pd.DataFrame([{"detector": "input (weather)", "model": "all", "day": str(d.date())} for d in inp_sig] +
             [{"detector": "error", "model": NAMES[k], "day": str(d.date())} for k in NAMES for d in sigs[k][1]]
             ).to_csv(os.path.join(OUT, "frozen_trigger_dates.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="frozen_trig_"))
import matplotlib; matplotlib.use("Agg")                        # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.lines import Line2D                             # noqa: E402
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#344054', 'text.color': '#182230',
                     'xtick.color': '#475467', 'ytick.color': '#475467', 'axes.edgecolor': '#D0D5DD', 'legend.frameon': False,
                     'savefig.facecolor': 'white'})
PLOTTED = ["demand_lightgbm63", "price_lightgbm31"]              # the figure shows the LightGBM models only (tables: all four)


def draw(error_ls, suffix):
    """Top panel: the input (weather) detector's statistic with its own signals. Then one panel per frozen model: its
    error with only its error-detector signals. error_ls: line style of the error-signal lines. No gridlines."""
    fig, axes = plt.subplots(1 + len(PLOTTED), 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [0.8] + [1] * len(PLOTTED)})
    ax = axes[0]
    stat = wscore.rolling(28).mean().loc[IDX]                 # = the detector statistic (window seeded with late 2014)
    for d in inp_sig:
        ax.axvline(d, color=SIG["input"], lw=1.0, alpha=.45, zorder=1)
    ax.plot(IDX, stat, color='#344054', lw=1.4, zorder=3)
    ax.axvline(pd.Timestamp("2025-01-01"), color='#344054', lw=1.1, ls=':', zorder=2)
    ax.set_title(f'Input detector (shared by all models): weather drift score, {len(inp_sig)} signals', loc='left')
    ax.set_ylabel("Weather score,\n28-day mean")
    for ax, k in zip(axes[1:], PLOTTED):
        err, err_sig, _ = sigs[k]
        for d in err_sig:
            ax.axvline(d, color=SIG["error"], lw=1.0, alpha=.45, ls=error_ls, zorder=1)
        ax.plot(IDX, err.rolling(28).mean(), color=LINE[k], lw=1.6, zorder=3)
        ax.axvline(pd.Timestamp("2025-01-01"), color='#344054', lw=1.1, ls=':', zorder=2)
        ax.set_title(f'Error detector · {NAMES[k]} — frozen (trained 2010–2014): {len(err_sig)} signals', loc='left')
        ax.set_ylabel("MAPE (%), 28-day mean" if k.startswith("demand") else "Error rate (%), 28-day mean")
    axes[-1].set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2026-11-01"))
    axes[-1].set_xlabel('Day (dotted line = start of 2025)')
    handles = [Line2D([], [], color='#344054', lw=1.4, label='Input detector statistic (28-day mean weather score)'),
               Line2D([], [], color=SIG["input"], lw=1.0, alpha=.6, label='Input detector signal (2014 reference)'),
               Line2D([], [], color=LINE[PLOTTED[0]], lw=1.6, label='Frozen model error (28-day mean)'),
               Line2D([], [], color=SIG["error"], lw=1.0, alpha=.6, ls=error_ls, label='Error detector signal (2014 reference)')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(.5, -.005))
    fig.tight_layout(rect=(0, .045, 1, 1))
    name = f"frozen_error_with_detector_signals{suffix}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("saved", os.path.join(OUT, name + ".png"))


draw((0, (4, 2)), "")          # error signals dashed
draw("-", "_solid")            # all signal lines solid
