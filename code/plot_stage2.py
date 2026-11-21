"""Stage 2 report figures: frozen degradation, drift detectors and adaptation strategies.

Reads only saved protocol outputs (no refitting):
  results/protocol/run_experiments/   baselines, frozen candidates, demand adaptation (LightGBM-63)
  results/protocol/price_c2/          C2 frozen and adapted (cut-off tuned on 2014 = 0.53)

Note: run_experiments' adapted_* PRICE columns belong to its own selection (LightGBM-31),
not C2, so price adaptation is always read from price_c2.

  python code/plot_stage2.py   ->  results/figures/stage2/*.pdf and *.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patches  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE = os.path.join(ROOT, "results", "protocol", "run_experiments")
C2 = os.path.join(ROOT, "results", "protocol", "price_c2")
OUT = os.path.join(ROOT, "results", "figures", "stage2")
CUT = "tuned_2014"
CHECK_START = pd.Timestamp("2025-01-01")  # 2025-26 kept aside as a later check

# Reference palette (dataviz skill): first three categorical slots, validated all-pairs in light mode.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"
WIDTH = 6.5  # inches, LaTeX text width

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def load():
    re = pd.read_csv(os.path.join(RE, "period_metrics.csv"))
    c2 = pd.read_csv(os.path.join(C2, "period_metrics.csv"))
    c2 = c2[c2.cutoff == CUT].drop(columns="cutoff")
    agg_re = pd.read_csv(os.path.join(RE, "aggregate_metrics.csv"))
    agg_c2 = pd.read_csv(os.path.join(C2, "aggregate_metrics.csv"))
    agg_c2 = agg_c2[agg_c2.cutoff == CUT].drop(columns="cutoff")
    signals_re = pd.read_csv(os.path.join(RE, "detector_signals.csv"), parse_dates=["day"])
    signals_c2 = pd.read_csv(os.path.join(C2, "detector_signals.csv"), parse_dates=["day"])
    signals_c2 = signals_c2[signals_c2.cutoff == CUT]
    retr_re = pd.read_csv(os.path.join(RE, "retrains.csv"), parse_dates=["effective_day"])
    retr_c2 = pd.read_csv(os.path.join(C2, "retrains.csv"), parse_dates=["effective_day"])
    retr_c2 = retr_c2[retr_c2.cutoff == CUT]
    return re, c2, agg_re, agg_c2, signals_re, signals_c2, retr_re, retr_c2


def monthly(frame, model):
    rows = frame[(frame.model == model) & (frame.frequency == "M")].copy()
    rows["date"] = pd.to_datetime(rows.period) + pd.Timedelta(days=14)
    return rows.set_index("date").sort_index()


def yearly(frame, model, column):
    rows = frame[(frame.model == model) & (frame.frequency == "Y")]
    return rows.set_index(rows.period.astype(int))[column].sort_index()


def agg(frame, model, period, column):
    return float(frame.loc[(frame.model == model) & (frame.period == period), column].iloc[0])


def time_axis(ax, label=True, label_y=1.0):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(pd.Timestamp("2014-12-01"), pd.Timestamp("2026-11-15"))
    ax.axvline(CHECK_START, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    if label:
        ax.text(CHECK_START, label_y, " 2025–26 check", transform=ax.get_xaxis_transform(),
                ha="left", va="top" if label_y > 0.5 else "bottom", fontsize=7, color=INK2)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".{pdf,png}"))


def band_plot(ax, rows, metric, color, label, iid=True):
    ax.fill_between(rows.index, rows[f"{metric}_day_boot_low"], rows[f"{metric}_day_boot_high"],
                    color=color, alpha=0.16, lw=0, label="95% CI, day-block bootstrap")
    if iid:
        ax.fill_between(rows.index, rows[f"{metric}_iid_low"], rows[f"{metric}_iid_high"],
                        color=color, alpha=0.40, lw=0, label="95% CI, half-hours i.i.d.")
    ax.plot(rows.index, rows[metric], color=color, lw=1.3, label=label)


# Figure 1: monthly error of the frozen selected models with both intervals ----------------------
def fig_frozen_monthly(re, c2):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(WIDTH, 4.8), sharex=True)
    base = monthly(re, "baseline_1d")
    top.plot(base.index, base.MAPE, color=INK2, lw=0.9, ls=(0, (3, 2)),
             label="Baseline: same half-hour yesterday")
    band_plot(top, monthly(re, "frozen_boosted_3"), "MAPE", BLUE, "Frozen LightGBM (63 leaves)")
    top.set_ylabel("Demand MAPE (%)")
    top.set_title("Demand", loc="left", fontweight="bold")
    top.legend(loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    time_axis(top)

    bottom.plot(base.index, base.accuracy, color=INK2, lw=0.9, ls=(0, (3, 2)),
                label="Baseline: same half-hour yesterday")
    band_plot(bottom, monthly(c2, "frozen_c2"), "accuracy", AQUA, "Frozen C2 (cut-off 0.53)")
    bottom.set_ylabel("Price-direction accuracy (%)")
    bottom.set_title("Price direction", loc="left", fontweight="bold")
    bottom.legend(loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    time_axis(bottom, label=False)
    bottom.set_xlabel("Month")
    fig.align_ylabels()
    fig.tight_layout(h_pad=1.5)
    save(fig, "fig_stage2_frozen_monthly")


# Figure 2: all frozen Stage 1 candidates against the baselines, by year -------------------------
DEMAND_SERIES = [  # model, label, colour, linestyle, width
    ("baseline_7d", "Baseline: last week", MUTED, ":", 1.1),
    ("baseline_1d", "Baseline: yesterday", INK2, "-", 1.1),
    ("frozen_linear_0", "Ridge α=10", ORANGE, "--", 1.1),
    ("frozen_linear_1", "Ridge α=1000", ORANGE, "-", 1.1),
    ("frozen_boosted_2", "LightGBM 31 leaves", BLUE, "--", 1.1),
    ("frozen_boosted_3", "LightGBM 63 leaves (selected)", BLUE, "-", 2.0),
]
PRICE_SERIES = [
    ("baseline_7d", "Baseline: last week", MUTED, ":", 1.1),
    ("baseline_1d", "Baseline: yesterday", INK2, "-", 1.1),
    ("frozen_linear_0", "Logistic C=0.01", ORANGE, "--", 1.1),
    ("frozen_linear_1", "Logistic C=0.1", ORANGE, "-", 1.1),
    ("frozen_boosted_2", "LightGBM 31 leaves", BLUE, "--", 1.1),
    ("frozen_boosted_3", "LightGBM 63 leaves", BLUE, "-", 1.1),
    ("frozen_c2", "C2 (selected)", AQUA, "-", 2.0),
]


def fig_frozen_yearly(re, c2):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), sharex=True)
    (d_abs, p_abs), (d_rel, p_rel) = axes
    d_base = yearly(re, "baseline_1d", "MAPE")
    p_base = yearly(re, "baseline_1d", "balanced_accuracy")
    handles = {}
    for model, label, color, ls, lw in DEMAND_SERIES:
        values = yearly(re, model, "MAPE")
        (line,) = d_abs.plot(values.index, values, color=color, ls=ls, lw=lw, marker="o", ms=2.5)
        handles[label] = line
        if not model.startswith("baseline"):
            d_rel.plot(values.index, values / d_base, color=color, ls=ls, lw=lw, marker="o", ms=2.5)
    for model, label, color, ls, lw in PRICE_SERIES:
        frame = c2 if model == "frozen_c2" else re
        values = yearly(frame, model, "balanced_accuracy")
        (line,) = p_abs.plot(values.index, values, color=color, ls=ls, lw=lw, marker="o", ms=2.5)
        handles.setdefault(label, line)
        if not model.startswith("baseline"):
            p_rel.plot(values.index, values - p_base, color=color, ls=ls, lw=lw, marker="o", ms=2.5)
    d_rel.axhline(1, color=INK2, lw=0.9)
    p_rel.axhline(0, color=INK2, lw=0.9)
    d_rel.text(2014.8, 1.02, "worse than baseline ↑", fontsize=7, color=INK2, va="bottom", ha="left")
    p_rel.text(2026.4, -0.6, "below baseline ↓", fontsize=7, color=INK2, va="top", ha="right")
    d_abs.set(ylabel="MAPE (%)", title="Demand: MAPE (lower is better)")
    p_abs.set(ylabel="Balanced accuracy (%)", title="Price direction: BA (higher is better)")
    d_rel.set(ylabel="MAPE ÷ baseline MAPE", xlabel="Year")
    p_rel.set(ylabel="BA − baseline BA (points)", xlabel="Year")
    for ax in axes.ravel():
        ax.title.set_fontweight("bold")
        ax.title.set_ha("left")
        ax.title.set_position((0, 1))
        ax.set_xticks(range(2015, 2027))
        ax.set_xticklabels([str(y)[2:] if y < 2026 else "26*" for y in range(2015, 2027)])
        ax.axvspan(2024.5, 2026.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    order = ["Baseline: yesterday", "Baseline: last week", "Ridge α=10", "Ridge α=1000",
             "Logistic C=0.01", "Logistic C=0.1", "LightGBM 31 leaves", "LightGBM 63 leaves",
             "LightGBM 63 leaves (selected)", "C2 (selected)"]
    fig.legend([handles[k] for k in order], order, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.07))
    fig.align_ylabels()
    fig.tight_layout()
    save(fig, "fig_stage2_frozen_yearly")


# Figure 3: drift detectors on the frozen models, with drift-triggered retrains ------------------
def detector_panel(ax, rows, column, threshold, signal, color, ylabel, retrains, title):
    ax.plot(rows.day, rows[column], color=color, lw=0.9, label="28-day rolling mean")
    ax.axhline(rows[threshold].iloc[0], color=INK, lw=0.9, ls=(0, (4, 2)),
               label="Threshold (99th pct. of 2014)")
    hits = rows.loc[rows[signal], "day"]
    ax.plot(hits, [1.0] * len(hits), transform=ax.get_xaxis_transform(), ls="none",
            marker="|", ms=6, mew=1.0, color=INK2, label="Signal")
    for i, day in enumerate(retrains):
        ax.axvline(day, color=ORANGE, lw=1.0, label="Drift-triggered retrain" if i == 0 else None)
    if len(retrains):
        ax.axvspan(retrains.max(), pd.Timestamp("2026-11-15"), color=GRID, alpha=0.5, lw=0,
                   zorder=0, label="Budget of 6 retrains used up" if title else None)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.14)
    ax.set_title(title, loc="left", fontweight="bold")
    above = (rows[column] > rows[threshold]).mean() * 100
    ax.set_title(f"above threshold on {above:.0f}% of days · {int(rows[signal].sum())} signals",
                 loc="right", fontsize=7.5, color=INK2)


def fig_detectors(signals_re, signals_c2, retr_re, retr_c2):
    demand = signals_re[(signals_re.task == "demand") & (signals_re.strategy == "frozen")]
    price = signals_c2[signals_c2.strategy == "frozen"]
    d_retr = retr_re[(retr_re.task == "demand") & (retr_re.strategy == "drift_sliding")].effective_day
    p_retr = retr_c2[retr_c2.strategy == "drift_sliding"].effective_day
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.4), sharex=True)
    detector_panel(axes[0], demand, "input_statistic", "input_threshold", "input_signal", INK2,
                   "Day-type drift score", pd.Series(dtype="datetime64[ns]"),
                   "Input detector (shared by both tasks)")
    detector_panel(axes[1], demand, "error_statistic", "error_threshold", "error_signal", BLUE,
                   "Daily MAPE (%)", d_retr, "Error detector: frozen demand LightGBM")
    detector_panel(axes[2], price, "error_statistic", "error_threshold", "error_signal", AQUA,
                   "Daily error rate (%)", p_retr, "Error detector: frozen price C2")
    for i, ax in enumerate(axes):
        time_axis(ax, label=(i == 0), label_y=0.02)
    handles, labels = [], []
    for ax in axes[:2]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels and lab != "28-day rolling mean":
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    axes[-1].set_xlabel("Day")
    fig.align_ylabels()
    fig.tight_layout()
    save(fig, "fig_stage2_drift_detectors")


# Figure 4: each adaptation strategy against the frozen model, monthly with CIs ------------------
STRATEGIES = [("periodic_expanding", "Periodic expanding (every 2 years, all data)"),
              ("periodic_sliding", "Periodic sliding (every 2 years, last 2 years)"),
              ("drift_sliding", "Drift-triggered (last 4 years, 180-day cool-down)")]


def fig_adaptation_monthly(re, c2, agg_re, agg_c2, retr_re, retr_c2):
    fig = plt.figure(figsize=(WIDTH, 7.6), layout="constrained")
    rows_fig = fig.subfigures(3, 1, hspace=0.04)
    axes = []
    for sub, (_, name) in zip(rows_fig, STRATEGIES):
        axes.append(sub.subplots(1, 2))
        sub.suptitle(name, x=0.0, ha="left", fontweight="bold", fontsize=8.5)
    import numpy as np  # local: only needed to index the grid like plt.subplots does
    axes = np.array(axes)
    for col in (0, 1):  # share x and y within each column, as sharey="col" did
        for ax in axes[1:, col]:
            ax.sharex(axes[0, col])
            ax.sharey(axes[0, col])
    frozen_d, frozen_p = monthly(re, "adapted_frozen"), monthly(c2, "frozen_c2")
    for row, (strategy, name) in enumerate(STRATEGIES):
        d_ax, p_ax = axes[row]
        d_rows, p_rows = monthly(re, f"adapted_{strategy}"), monthly(c2, f"adapted_c2_{strategy}")
        for ax, frozen, rows, metric, color in ((d_ax, frozen_d, d_rows, "MAPE", BLUE),
                                                (p_ax, frozen_p, p_rows, "accuracy", AQUA)):
            ax.plot(frozen.index, frozen[metric], color=MUTED, lw=0.9, label="Frozen")
            band_plot(ax, rows, metric, color, "Adapted", iid=False)
        days_d = retr_re[(retr_re.task == "demand") & (retr_re.strategy == strategy)].effective_day
        days_p = retr_c2[retr_c2.strategy == strategy].effective_day
        for ax, days in ((d_ax, days_d), (p_ax, days_p)):
            ax.plot(days, [0.0] * len(days), transform=ax.get_xaxis_transform(), ls="none",
                    marker="^", ms=5, color=ORANGE, clip_on=False, label="Retrain")
            time_axis(ax, label=False)
        d_txt = (f"MAPE 2015–24: {agg(agg_re, f'adapted_{strategy}', 'strategy_selection', 'MAPE'):.2f}% "
                 f"(frozen {agg(agg_re, 'adapted_frozen', 'strategy_selection', 'MAPE'):.2f})\n"
                 f"MAPE 2025–26: {agg(agg_re, f'adapted_{strategy}', 'holdout', 'MAPE'):.2f}% "
                 f"(frozen {agg(agg_re, 'adapted_frozen', 'holdout', 'MAPE'):.2f})")
        p_txt = (f"BA 2015–24: {agg(agg_c2, f'adapted_c2_{strategy}', 'strategy_selection', 'balanced_accuracy'):.1f} "
                 f"(frozen {agg(agg_c2, 'frozen_c2', 'strategy_selection', 'balanced_accuracy'):.1f})\n"
                 f"BA 2025–26: {agg(agg_c2, f'adapted_c2_{strategy}', 'holdout', 'balanced_accuracy'):.1f} "
                 f"(frozen {agg(agg_c2, 'frozen_c2', 'holdout', 'balanced_accuracy'):.1f})")
        d_ax.set_title(d_txt, loc="left", fontsize=7, color=INK2)
        p_ax.set_title(p_txt, loc="left", fontsize=7, color=INK2)
        d_ax.set_ylabel("Demand MAPE (%)")
        p_ax.set_ylabel("Price accuracy (%)")
    for ax in axes[-1]:
        ax.set_xlabel("Month")
    handles = [plt.Line2D([], [], color=MUTED, lw=0.9),
               plt.Line2D([], [], color=BLUE, lw=1.3), plt.Line2D([], [], color=AQUA, lw=1.3),
               matplotlib.patches.Patch(color=INK2, alpha=0.16, lw=0),
               plt.Line2D([], [], ls="none", marker="^", ms=5, color=ORANGE)]
    labels = ["Frozen", "Adapted: demand LightGBM", "Adapted: price C2",
              "95% CI of adapted model (day-block bootstrap)", "Retrain takes effect"]
    for ax in axes[:-1].ravel():
        ax.tick_params(labelbottom=False)
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    save(fig, "fig_stage2_adaptation_monthly")


# Figure 5: summary of strategies (replaces the adaptation table) -------------------------------
def fig_adaptation_summary(agg_re, agg_c2):
    rows = [("Frozen (0)", "adapted_frozen", "frozen_c2"),
            ("Periodic expanding (6)", "adapted_periodic_expanding", "adapted_c2_periodic_expanding"),
            ("Periodic sliding (6)", "adapted_periodic_sliding", "adapted_c2_periodic_sliding"),
            ("Drift-triggered (6)", "adapted_drift_sliding", "adapted_c2_drift_sliding")]
    fig, (d_ax, p_ax) = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    y = list(range(len(rows)))[::-1]
    for (label, d_model, p_model), yy in zip(rows, y):
        for ax, frame, model, metric, fmt in ((d_ax, agg_re, d_model, "MAPE", "{:.2f}"),
                                              (p_ax, agg_c2, p_model, "balanced_accuracy", "{:.1f}")):
            a = agg(frame, model, "strategy_selection", metric)
            b = agg(frame, model, "holdout", metric)
            ax.plot([a, b], [yy, yy], color=GRID, lw=2, zorder=1)
            ax.plot(a, yy, "o", color=BLUE, ms=6, zorder=3, label="2015–2024")
            ax.plot(b, yy, "D", color=ORANGE, ms=5, zorder=3, label="2025–Oct 2026")
            for value, va, dy in ((a, "top", -0.14), (b, "bottom", 0.14)):
                ax.text(value, yy + dy, fmt.format(value), ha="center", va=va, fontsize=7,
                        color=INK2)
    for ax, frame, metric in ((d_ax, agg_re, "MAPE"), (p_ax, agg_re, "balanced_accuracy")):
        for period, ls in (("strategy_selection", "-"), ("holdout", "--")):
            ax.axvline(agg(frame, "baseline_1d", period, metric), color=MUTED, lw=0.9, ls=ls)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.7, len(rows) - 0.3)
    d_ax.set_yticks(y)
    d_ax.set_yticklabels([r[0] for r in rows])
    d_ax.set_xlabel("Demand MAPE (%), lower is better")
    p_ax.set_xlabel("Price balanced accuracy (%), higher is better")
    d_ax.set_title("Demand: LightGBM 63 leaves", loc="left", fontweight="bold")
    p_ax.set_title("Price: C2 (cut-off 0.53)", loc="left", fontweight="bold")
    handles = d_ax.get_legend_handles_labels()[0][:2] + [
        plt.Line2D([], [], color=MUTED, lw=0.9), plt.Line2D([], [], color=MUTED, lw=0.9, ls="--")]
    labels = ["2015–2024", "2025–Oct 2026",
              "Yesterday baseline, 2015–24", "Yesterday baseline, 2025–26"]
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_stage2_adaptation_summary")


def main():
    re, c2, agg_re, agg_c2, signals_re, signals_c2, retr_re, retr_c2 = load()
    fig_frozen_monthly(re, c2)
    fig_frozen_yearly(re, c2)
    fig_detectors(signals_re, signals_c2, retr_re, retr_c2)
    fig_adaptation_monthly(re, c2, agg_re, agg_c2, retr_re, retr_c2)
    fig_adaptation_summary(agg_re, agg_c2)


if __name__ == "__main__":
    main()
