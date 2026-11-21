"""Demand MAPE by day type (previous vs adopted model) and the adopted model's MAPE by type and year, 2014-2026
(2026 = Jan-Oct). Day types: team k-means fitted on 2010-2013 (fixed). Models (50/50 blends, 405 features):
previous = one LightGBM + one ten-epoch MLP; adopted = one LightGBM + mean of five MLPs (main 783e503).
Forecasts, all saved on origin/main (read from the checkout if present, else via git show):
  2014: results/demand_seed_averaging_5seeds_2014/predictions.csv (trained <= 2013; columns current, bagged_MLP_only)
  2015-2024: results/demand_seed_averaging_5seeds_2015_2024/predictions.csv (biennial walk-forward)
  2025-2026: results/demand_single_lgbm_5mlp_holdout/predictions.csv (2025 trained <= 2023, 2026 trained <= 2025)
Descriptive figures, not model selection. Run from the repo root."""
import io, os, subprocess
import numpy as np, pandas as pd
from common import team_day_types, load_demand, pyplot, save, ROOT, OUT, COLOURS, END


def read_main(rel, cols, rename):
    path = os.path.join(ROOT, rel)
    src = path if os.path.exists(path) else io.BytesIO(subprocess.run(
        ["git", "-C", ROOT, "show", f"origin/main:{rel}"], capture_output=True, check=True).stdout)
    return pd.read_csv(src, usecols=cols, index_col="timestamp", parse_dates=True).rename(columns=rename)


fc = pd.concat([
    read_main("results/demand_seed_averaging_5seeds_2014/predictions.csv", ["timestamp", "actual", "current", "bagged_MLP_only"], {"bagged_MLP_only": "adopted"}),
    read_main("results/demand_seed_averaging_5seeds_2015_2024/predictions.csv", ["timestamp", "actual", "current", "bagged_MLP_only_5"], {"bagged_MLP_only_5": "adopted"}),
    read_main("results/demand_single_lgbm_5mlp_holdout/predictions.csv", ["timestamp", "actual", "current", "candidate"], {"candidate": "adopted"})])
assert fc.index.is_unique and fc.index.min() == pd.Timestamp("2014-01-01 00:30") and fc.index.max() == pd.Timestamp("2026-11-01 00:00")
assert len(fc) == ((pd.Timestamp(END) - pd.Timestamp("2014-01-01")).days + 1) * 48
np.testing.assert_array_equal(fc["actual"].to_numpy(), load_demand(END).loc[fc.index, "nsw_demand"].to_numpy())

types, _, _, _ = team_day_types(END)
t = types.loc[types.index.year >= 2014].copy()
truth = fc["actual"].to_numpy().reshape(-1, 48)
for m in ("current", "adopted"):
    t[m] = 100 * np.mean(np.abs(truth - fc[m].to_numpy().reshape(-1, 48)) / truth, axis=1)
by_type = t.groupby("cluster")[["current", "adopted"]].mean()
yearly = t.groupby(["year", "cluster"]).agg(days=("cluster", "size"), MAPE=("adopted", "mean")).reset_index()
os.makedirs(OUT, exist_ok=True)
by_type.to_csv(os.path.join(OUT, "mape_by_type_2014_2026.csv"))
yearly.to_csv(os.path.join(OUT, "mape_by_type_and_year_2014_2026.csv"), index=False)
print(by_type.round(4).to_string()); print(yearly.round(3).to_string(index=False))

plt = pyplot()
# MAPE by type, previous vs adopted
fig, ax = plt.subplots(figsize=(8, 4.8)); pos = np.arange(2); width = .33
for i, (m, label, colour) in enumerate((("current", "Previous: one LightGBM + one MLP", '#ABB3BE'),
                                        ("adopted", "Adopted: one LightGBM + five MLPs", '#D47735'))):
    bars = ax.bar(pos + (i - .5) * width, by_type[m], width, color=colour, label=label)
    ax.bar_label(bars, fmt='%.3f%%', padding=4, fontsize=10)
ax.set_xticks(pos, ['Type 0', 'Type 1'])
ax.set(title='Demand error by day type, 2014–2026', ylabel='MAPE (%) · lower is better', ylim=(0, float(by_type.max().max()) * 1.25))
ax.legend(loc='upper center', ncol=2, fontsize=9)
ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "mape_by_type_2014_2026")
# adopted model's MAPE by type and year
fig, ax = plt.subplots(figsize=(8, 4.6))
for k in range(2):
    rows = yearly.loc[yearly.cluster == k]
    ax.plot(rows.year, rows.MAPE, color=COLOURS[k], lw=2.2, marker='o', ms=4, label=f'Type {k}')
top = float(yearly.MAPE.max()) * 1.12
for _, r in yearly[yearly.cluster == 1].iterrows():
    ax.text(r.year, top, f'{int(r.days)}', ha='center', va='top', fontsize=7.5, color=COLOURS[1])
ax.text(2013.45, top, 'Type 1 days:', ha='right', va='top', fontsize=7.5, color=COLOURS[1])
ax.set(title='Adopted model error through time', xlabel='Year (2026 = Jan–Oct)', ylabel='MAPE (%) · lower is better',
       ylim=(None, top * 1.01), xlim=(2012.2, 2026.6))
ax.set_xticks([2014, 2016, 2018, 2020, 2022, 2024, 2026]); ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.88))
ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "mape_by_type_and_year_2014_2026")
