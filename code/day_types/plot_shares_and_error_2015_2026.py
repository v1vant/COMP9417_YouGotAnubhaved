"""Panel B and the B+D pair of code/refresh_current_day_types.py (origin/main), extended to 2015-2026 (2026 = Jan-Oct):
B: share of days assigned to each day type; C: demand MAPE by type, previous vs adopted model;
D: adopted model's MAPE by type and year. Day types: team k-means fitted on 2010-2013 (fixed).
Models (both 50/50 blends on 405 features):
  previous = one LightGBM + one ten-epoch MLP ('current'); adopted = one LightGBM + mean of five MLPs (main 783e503).
Forecasts, all saved on origin/main:
  2015-2024: results/demand_seed_averaging_5seeds_2015_2024/predictions.csv (columns current, bagged_MLP_only_5)
  2025-2026: results/demand_single_lgbm_5mlp_holdout/predictions.csv (columns current, candidate = the adopted blend);
             2025 forecast by models trained to 2023-12-31, 2026 by models trained to 2025-12-31 (frozen_plan.json).
Files are read from the checkout if present, else via git show. Descriptive figures, not model selection."""
import io, os, subprocess
import numpy as np, pandas as pd
from common import team_day_types, load_demand, pyplot, save, ROOT, OUT, COLOURS, END


def read_main(rel, cols):
    path = os.path.join(ROOT, rel)
    src = path if os.path.exists(path) else io.BytesIO(subprocess.run(
        ["git", "-C", ROOT, "show", f"origin/main:{rel}"], capture_output=True, check=True).stdout)
    return pd.read_csv(src, usecols=cols, index_col="timestamp", parse_dates=True)


a = read_main("results/demand_seed_averaging_5seeds_2015_2024/predictions.csv", ["timestamp", "actual", "current", "bagged_MLP_only_5"])
b = read_main("results/demand_single_lgbm_5mlp_holdout/predictions.csv", ["timestamp", "actual", "current", "candidate"])
fc = pd.concat([a.rename(columns={"bagged_MLP_only_5": "adopted"}), b.rename(columns={"candidate": "adopted"})])
assert fc.index.is_unique and fc.index.min() == pd.Timestamp("2015-01-01 00:30") and fc.index.max() == pd.Timestamp("2026-11-01 00:00")
assert len(fc) == (pd.Timestamp(END) - pd.Timestamp("2015-01-01")).days * 48 + 48
np.testing.assert_array_equal(fc["actual"].to_numpy(), load_demand(END).loc[fc.index, "nsw_demand"].to_numpy())

types, _, _, _ = team_day_types(END)
t = types.loc[types.index.year >= 2015].copy()
truth = fc["actual"].to_numpy().reshape(-1, 48)
for m in ("current", "adopted"):
    t[m] = 100 * np.mean(np.abs(truth - fc[m].to_numpy().reshape(-1, 48)) / truth, axis=1)
assert (np.bincount(t.loc[t.index.year <= 2024, "cluster"]) == [2927, 726]).all()   # 2015-2024 as Anubhav's figure
shares = pd.crosstab(t["year"], t["cluster"], normalize="index")
by_type = t.groupby("cluster")[["current", "adopted"]].mean()
yearly = t.groupby(["year", "cluster"]).agg(days=("cluster", "size"), MAPE=("adopted", "mean")).reset_index()
os.makedirs(OUT, exist_ok=True)
shares.to_csv(os.path.join(OUT, "day_type_shares_2015_2026.csv"))
print(shares.round(3).to_string()); print(by_type.round(4).to_string()); print(yearly.round(3).to_string(index=False))

plt = pyplot()
TICKS = [2015, 2017, 2019, 2021, 2023, 2025, 2026]


def panel_b(ax):
    ax.stackplot(shares.index, *[100 * shares[k] for k in range(2)], colors=COLOURS, alpha=.85, labels=['Type 0', 'Type 1'])
    ax.set(title='Share of days assigned to each type', xlabel='Year (2026 = Jan–Oct)', ylabel='Share of days (%)',
           ylim=(0, 100), xlim=(2015, 2026))
    ax.set_xticks(TICKS); ax.legend(loc='lower left', ncol=2, facecolor='white')


def panel_c(ax):
    pos = np.arange(2); width = .33
    for i, (m, label, colour) in enumerate((("current", "Previous: one LightGBM + one MLP", '#ABB3BE'),
                                            ("adopted", "Adopted: one LightGBM + five MLPs", '#D47735'))):
        bars = ax.bar(pos + (i - .5) * width, by_type[m], width, color=colour, label=label)
        ax.bar_label(bars, fmt='%.3f%%', padding=4, fontsize=10)
    ax.set_xticks(pos, ['Type 0', 'Type 1'])
    ax.set(title='Demand error by day type, 2015–2026', ylabel='MAPE (%) · lower is better', ylim=(0, float(by_type.max().max()) * 1.25))
    ax.legend(loc='upper center', ncol=2, fontsize=9)


def panel_d(ax):
    for k in range(2):
        rows = yearly.loc[yearly.cluster == k]
        ax.plot(rows.year, rows.MAPE, color=COLOURS[k], lw=2.2, marker='o', ms=4, label=f'Type {k}')
    for yr in (2024, 2025, 2026):
        r = yearly.loc[(yearly.year == yr) & (yearly.cluster == 1)]
        if len(r) and r.iloc[0].days < 20:
            ax.annotate(f'{int(r.iloc[0].days)} days', xy=(yr, r.iloc[0].MAPE), xytext=(0, 7), textcoords='offset points',
                        ha='center', fontsize=8, color=COLOURS[1])
    ax.set(title='Adopted model error through time', xlabel='Year (2026 = Jan–Oct)', ylabel='MAPE (%) · lower is better')
    ax.set_xticks(TICKS); ax.legend(loc='upper left')


def tidy(ax):
    ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(14, 4.8)); panel_b(axes[0]); panel_d(axes[1]); [tidy(x) for x in axes]
fig.tight_layout(); save(fig, "day_type_shares_and_error_2015_2026")
# single-panel C and D now come from plot_mape_by_type_2014_2026.py
for name, fn, size in (("day_type_shares_2015_2026", panel_b, (7.5, 4.6)),):
    fig, ax = plt.subplots(figsize=size); fn(ax); tidy(ax); fig.tight_layout(); save(fig, name)
