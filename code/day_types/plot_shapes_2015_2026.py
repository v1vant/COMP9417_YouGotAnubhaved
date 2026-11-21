"""Panel A of code/refresh_current_day_types.py on its own: mean 2015-2026 shape of the days assigned to each type
(solid) against the fixed 2010-2013 centroid (dashed). Run from the repo root."""
import numpy as np
from common import team_day_types, load_demand, pyplot, save, COLOURS, END

types, _, typical, _ = team_day_types(END)
frame = load_demand(END)
keep = np.asarray(types.index.year >= 2015)
truth = frame["nsw_demand"].to_numpy().reshape(-1, 48)[keep]
profiles = truth / truth.mean(axis=1, keepdims=True)
labels = types["cluster"].to_numpy()[keep]
upto24 = np.asarray(types.index.year[keep] <= 2024)
assert (np.bincount(labels[upto24]) == [2927, 726]).all()  # 2015-2024 days as results/day_types_current_mlp5_2015_2024
actual = np.array([profiles[labels == k].mean(axis=0) for k in range(2)])
plt = pyplot()
fig, ax = plt.subplots(figsize=(7.5, 4.6)); hours = np.arange(48) / 2
for k in range(2):
    ax.plot(hours, actual[k], color=COLOURS[k], lw=2.5, label=f'Type {k} · {int((labels == k).sum()):,} days')
    ax.plot(hours, typical[k], color=COLOURS[k], lw=1.4, ls='--', alpha=.65)
ax.axhline(1, color='#98A2B3', lw=.8, zorder=0)
ax.set(title='Demand shape within each type', xlabel='Interval start (AEST)', ylabel='Demand / daily mean',
       xlim=(0, 24), ylim=(.68, 1.26))
ax.set_xticks([0, 6, 12, 18, 24], ['00:00', '06:00', '12:00', '18:00', '24:00'])
ax.legend(loc='upper left', fontsize=9)
ax.text(.02, .025, 'Solid: 2015–2026 mean   Dashed: fixed 2010–2013 centroid', transform=ax.transAxes, fontsize=8.5, color='#667085')
ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "day_type_shapes_2015_2026")
