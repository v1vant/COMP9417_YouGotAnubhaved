"""Plots and tables for simulate_drift.py: model error (28-day rolling mean of daily error, i.e. the error detector's
statistic) under input-triggered and error-triggered retraining, against the frozen model, with detector signals and
the days retrains take effect. Run from the repo root after simulate_drift.py."""
import json, os, tempfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # plots/ -> repo root
OUT = os.path.join(ROOT, "results", "drift_detectors_2015_2026")
NAMES = {"demand_lightgbm63": "Demand · LightGBM (63 leaves)", "demand_ridge1000": "Demand · Ridge (α = 1000)",
         "price_lightgbm31": "Price · LightGBM (31 leaves), cut-off 0.5", "price_logistic001": "Price · Logistic (C = 0.01), cut-off 0.5"}
COL = {"input": "#7F56D9", "error": "#E5732E"}
R = {k: json.load(open(os.path.join(OUT, f"{k}.json"))) for k in NAMES}

# ---- setup and summary tables
setup, summary, retr = [], [], []
for k, r in R.items():
    unit = "MAPE (%)" if r["task"] == "demand" else "error rate (% of half-hours wrong)"
    setup.append({"model": NAMES[k], "detector": "input",
                  "monitored statistic": "28-day rolling mean of the daily day-type drift score",
                  "reference period": "2014 (day types fitted on 2010-2013)",
                  "threshold": round(r["input"]["input_threshold"], 4),
                  "how chosen": "99th percentile of the 28-day rolling means over 2014"})
    setup.append({"model": NAMES[k], "detector": "error",
                  "monitored statistic": f"28-day rolling mean of the deployed model's daily {unit}",
                  "reference period": "2014: daily errors of the model fitted on 2010-2013 (Step 1 validation year)",
                  "threshold": round(r["error"]["error_threshold"], 4),
                  "how chosen": "99th percentile of the 28-day rolling means over 2014"})
    fz = pd.Series(r["frozen_daily_error"], index=pd.date_range("2015-01-01", "2026-10-31"))
    for trig in ("input", "error"):
        L = pd.DataFrame(r[trig]["log"]); L["day"] = pd.to_datetime(L.day); L = L.set_index("day")
        sig = L[f"{trig}_signal"]
        for p, m in (("2015-24", L.index.year <= 2024), ("2025-26", L.index.year >= 2025)):
            summary.append({"model": NAMES[k], "policy": f"{trig}-triggered", "period": p,
                            "signals of the triggering detector": int(sig[m].sum()),
                            "retrains effective": sum(1 for x in r[trig]["retrains"] if (pd.Timestamp(x["effective_day"]).year <= 2024) == (p == "2015-24")),
                            "mean daily error": round(L.error[m].mean(), 3), "frozen mean daily error": round(fz[m].mean(), 3)})
        for x in r[trig]["retrains"]:
            retr.append({"model": NAMES[k], "policy": f"{trig}-triggered", **x})
pd.DataFrame(setup).to_csv(os.path.join(OUT, "detector_setup.csv"), index=False)
S = pd.DataFrame(summary); S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
pd.DataFrame(retr).to_csv(os.path.join(OUT, "retrains.csv"), index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(setup)[["model", "detector", "threshold"]].to_string(index=False)); print(S.to_string(index=False))
print(pd.DataFrame(retr)[["model", "policy", "effective_day", "training_start", "training_end"]].to_string(index=False))

# ---- figures
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="drift_"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#344054', 'text.color': '#182230',
                     'xtick.color': '#475467', 'ytick.color': '#475467', 'axes.edgecolor': '#D0D5DD', 'legend.frameon': False,
                     'savefig.facecolor': 'white'})


def figure(task, fname, ylabel):
    keys = [k for k in NAMES if R[k]["task"] == task]
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    for row, (k, trig) in enumerate([(k, t) for k in keys for t in ("input", "error")]):
        ax, r = axes[row], R[k]
        L = pd.DataFrame(r[trig]["log"]); L["day"] = pd.to_datetime(L.day); L = L.set_index("day")
        fz = pd.Series(r["frozen_daily_error"], index=L.index).rolling(28).mean()
        ax.plot(fz.index, fz.values, color='#98A2B3', lw=1.1, label='Frozen model (trained 2010–2014)')
        ax.plot(L.index, L.error.rolling(28).mean(), color=COL[trig], lw=1.5,
                label=f'Model retrained when the {trig} detector signals')
        thr = r[trig]["error_threshold"]
        if trig == "error":                                   # threshold only where the error detector triggers
            ax.axhline(thr, color='#344054', lw=1, ls='--', label='Error-detector threshold (99th pct of 2014)')
        lo, hi = ax.get_ylim(); top = hi * 1.0
        for j, det in enumerate(("input", "error")):              # signal ticks for both detectors
            d = L.index[L[f"{det}_signal"]]
            y = hi + (hi - lo) * (0.05 + 0.06 * j)
            ax.scatter(d, [y] * len(d), marker='|', s=40, color=COL[det], alpha=1 if det == trig else .45, clip_on=False)
            ax.text(pd.Timestamp("2014-12-20"), y, f'{det} signals', ha='right', va='center', fontsize=7.5, color=COL[det])
        for x in r[trig]["retrains"]:                               # retrains take effect
            t = pd.Timestamp(x["effective_day"])
            ax.axvline(t, color=COL[trig], lw=1, alpha=.5)
            ax.scatter([t], [lo + (hi - lo) * 0.03], marker='^', s=60, color=COL[trig], edgecolor='white', zorder=5)
        ax.set_ylim(lo, hi)
        ax.axvline(pd.Timestamp("2025-01-01"), color='#667085', lw=.8, ls=':')
        n = len(r[trig]["retrains"])
        ax.set_title(f'{NAMES[k]} — {trig}-triggered retraining ({n} retrain{"s" if n != 1 else ""}, budget 6)', loc='left', pad=40)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
    axes[-1].set_xlabel('Day (28-day rolling mean of daily error; dotted line = start of 2025)')
    axes[-1].set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2026-11-01"))
    handles = [Line2D([], [], color='#98A2B3', lw=1.1, label='Frozen model (trained 2010–2014)'),
               Line2D([], [], color=COL["input"], lw=1.5, label='Retrained when the input detector signals'),
               Line2D([], [], color=COL["error"], lw=1.5, label='Retrained when the error detector signals'),
               Line2D([], [], color='#344054', lw=1, ls='--', label='Error-detector threshold (99th pct of 2014)'),
               Line2D([], [], color='#475467', marker='|', ls='', ms=10, label='Detector signal (input / error; faint = not triggering)'),
               Line2D([], [], color='#475467', marker='^', ls='', ms=8, label='Retrain takes effect (next day)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.5, bbox_to_anchor=(.5, -.005))
    fig.tight_layout(rect=(0, .04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=200)
    print("saved", os.path.join(OUT, fname + ".png"))


figure("demand", "drift_detectors_demand", "MAPE (%), 28-day mean")
figure("price", "drift_detectors_price", "Error rate (%), 28-day mean")
