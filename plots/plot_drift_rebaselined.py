"""Plots and tables for simulate_drift_rebaselined.py, compared with the original (fixed 2014 baseline) run.
Run from the repo root after both simulations."""
import itertools, json, os, tempfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # plots/ -> repo root
BASE = os.path.join(ROOT, "results", "drift_detectors_2015_2026")
INPUT_Q = float(os.environ.get("INPUT_QUANTILE", "0.99"))
OUT = os.path.join(BASE, "rebaselined" if INPUT_Q == 0.99 else f"rebaselined_input{INPUT_Q * 100:g}")
TAG = "" if INPUT_Q == 0.99 else f"_input{INPUT_Q * 100:g}"
IQ = f"{INPUT_Q * 100:g}th"
NAMES = {"demand_lightgbm63": "Demand · LightGBM (63 leaves)", "demand_ridge1000": "Demand · Ridge (α = 1000)",
         "price_lightgbm31": "Price · LightGBM (31 leaves), cut-off 0.5", "price_logistic001": "Price · Logistic (C = 0.01), cut-off 0.5"}
COL = {"input": "#7F56D9", "error": "#E5732E"}
R = {k: json.load(open(os.path.join(OUT, f"{k}.json"))) for k in NAMES}
O = {k: json.load(open(os.path.join(BASE, f"{k}.json"))) for k in NAMES}
IDX = pd.date_range("2015-01-01", "2026-10-31")

bl, summ = [], []
for k, r in R.items():
    fz = pd.Series(r["frozen_daily_error"], index=IDX)
    for trig in ("input", "error"):
        for b in r[trig]["baselines"]:
            bl.append({"model": NAMES[k], "policy": f"{trig}-triggered", "effective from": b["effective_day"],
                       "detectors fitted on": b["fit"], "reference": b["reference"],
                       "input threshold": round(b["input_threshold"], 4), "error threshold": round(b["error_threshold"], 4)})
        for variant, src in (("re-baselined", r), ("fixed 2014 baseline", O[k])):
            L = pd.Series([x["error"] for x in src[trig]["log"]], index=IDX)
            sig = pd.Series([x[f"{trig}_signal"] for x in src[trig]["log"]], index=IDX)
            dates = [x["effective_day"] for x in src[trig]["retrains"]]
            for p, m in (("2015-24", IDX.year <= 2024), ("2025-26", IDX.year >= 2025)):
                summ.append({"model": NAMES[k], "policy": f"{trig}-triggered", "variant": variant, "period": p,
                             "signals": int(sig[m].sum()), "mean daily error": round(L[m].mean(), 3),
                             "frozen": round(fz[m].mean(), 3),
                             "retrain dates": ", ".join(dates) if p == "2015-24" else ""})
B = pd.DataFrame(bl); B.to_csv(os.path.join(OUT, "baselines.csv"), index=False)
S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, "summary_vs_fixed_baseline.csv"), index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 120)
print(B.to_string(index=False)); print(S.to_string(index=False))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="drift_rb_"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': '#344054', 'text.color': '#182230',
                     'xtick.color': '#475467', 'ytick.color': '#475467', 'axes.edgecolor': '#D0D5DD', 'legend.frameon': False,
                     'savefig.facecolor': 'white'})


def figure(task, fname, ylabel):
    keys = [k for k in NAMES if R[k]["task"] == task]
    fig, axes = plt.subplots(2, 2, figsize=(22, 9.5), sharex=True)   # rows: models; columns: input / error
    for (row, k), (col, trig) in itertools.product(enumerate(keys), enumerate(("input", "error"))):
        ax, r = axes[row, col], R[k]
        L = pd.DataFrame(r[trig]["log"]); L.index = IDX
        fz = pd.Series(r["frozen_daily_error"], index=IDX).rolling(28).mean()
        ax.plot(fz.index, fz.values, color='#98A2B3', lw=1.1)
        ax.plot(IDX, L.error.rolling(28).mean(), color=COL[trig], lw=1.5)
        if trig == "error":                                   # threshold only where the error detector triggers
            ax.step(IDX, L.error_threshold, where='post', color='#344054', lw=1, ls='--')
        lo, hi = ax.get_ylim()
        for j, det in enumerate(("input", "error")):
            d = IDX[L[f"{det}_signal"].to_numpy()]
            y = hi + (hi - lo) * (0.05 + 0.06 * j)
            ax.scatter(d, [y] * len(d), marker='|', s=40, color=COL[det], alpha=1 if det == trig else .45, clip_on=False)
            ax.text(pd.Timestamp("2014-12-20"), y, f'{det} signals', ha='right', va='center', fontsize=7.5, color=COL[det])
        for x in r[trig]["retrains"]:
            t = pd.Timestamp(x["effective_day"])
            ax.axvline(t, color=COL[trig], lw=1, alpha=.5)
            ax.scatter([t], [lo + (hi - lo) * 0.03], marker='^', s=60, color=COL[trig], edgecolor='white', zorder=5)
        ax.set_ylim(lo, hi)
        ax.axvline(pd.Timestamp("2025-01-01"), color='#667085', lw=.8, ls=':')
        n = len(r[trig]["retrains"])
        ax.set_title(f'{NAMES[k]} — {trig}-triggered, re-baselined ({n} retrain{"s" if n != 1 else ""}, budget 6)',
                     loc='left', pad=40)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel('Day (28-day rolling mean of daily error; dotted line = start of 2025)')
    axes[-1, 0].set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2026-11-01"))
    handles = [Line2D([], [], color='#98A2B3', lw=1.1, label='Frozen model (trained 2010–2014)'),
               Line2D([], [], color=COL["input"], lw=1.5, label='Retrained when the input detector signals'),
               Line2D([], [], color=COL["error"], lw=1.5, label='Retrained when the error detector signals'),
               Line2D([], [], color='#344054', lw=1, ls='--', label='Error-detector threshold (99th pct of the latest reference year)'),
               Line2D([], [], color='#475467', marker='|', ls='', ms=10, label='Detector signal (input / error; faint = not triggering)'),
               Line2D([], [], color='#475467', marker='^', ls='', ms=8, label='Retrain + re-baseline take effect (next day)')]
    fig.suptitle(f'Re-baselined drift detectors — input (weather) threshold: {IQ} pct, error threshold: 99th pct',
                 x=.01, ha='left', fontsize=11.5, fontweight='bold')
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8.5, bbox_to_anchor=(.5, -.005))
    fig.tight_layout(rect=(0, .035, 1, 1), w_pad=4)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), dpi=200)
    print("saved", os.path.join(OUT, fname + ".png"))


figure("demand", f"drift_detectors_rebaselined{TAG}_demand", "MAPE (%), 28-day mean")
figure("price", f"drift_detectors_rebaselined{TAG}_price", "Error rate (%), 28-day mean")
