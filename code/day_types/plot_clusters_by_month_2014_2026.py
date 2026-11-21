"""Day type (team k-means, k=2, PCA + k-means fitted on 2010-2013 and kept fixed) of every day from January 2014 to
October 2026, counted by month. Light dashed lines: NSW daylight-saving changes (first Sunday of October / April).
Descriptive figure (includes 2025-2026 at the user's request). Run from the repo root."""
import os
import numpy as np, pandas as pd
from common import team_day_types, pyplot, save, OUT, COLOURS, NAMES, END

types, _, _, _ = team_day_types(END)
t = types.reset_index(); t = t[t["day"] >= "2014-01-01"]; t["month"] = t["day"].dt.to_period("M")
counts = pd.crosstab(t["month"], t["cluster"]).reindex(columns=[0, 1], fill_value=0)
os.makedirs(OUT, exist_ok=True); counts.to_csv(os.path.join(OUT, "clusters_by_month_2014_2026.csv"))
print(counts.groupby(counts.index.year).sum().to_string())


def first_sunday(y, m):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(6 - d.weekday()) % 7)


pos = lambda ts: (ts.year - 2014) * 12 + (ts.month - 1) + (ts.day - 1) / ts.days_in_month - 0.5
changes = [first_sunday(y, m) for y in range(2014, 2027) for m in (4, 10)]
months = counts.index; x = np.arange(len(months))
plt = pyplot()
fig, ax = plt.subplots(figsize=(15, 4.8))
ax.bar(x, counts[0], width=.8, color=COLOURS[0], edgecolor='white', linewidth=.4, label=NAMES[0])
ax.bar(x, counts[1], width=.8, bottom=counts[0], color=COLOURS[1], edgecolor='white', linewidth=.4, label=NAMES[1])
for ts in changes:
    ax.axvline(pos(ts), color='#98A2B3', lw=.7, ls='--', zorder=0)
jan = [i for i, m in enumerate(months) if m.month == 1]
ax.set_xticks(jan, [str(months[i].year) for i in jan])
ax.set(ylabel='Days in month', ylim=(0, 32), xlim=(-.7, len(months) - .3), xlabel='Month (to October 2026)',
       title='Day type by month, 2014–2026 (clusters fitted on 2010–2013)')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=3, fontsize=9,
          handles=ax.get_legend_handles_labels()[0] + [plt.Line2D([], [], color='#98A2B3', lw=.7, ls='--')],
          labels=ax.get_legend_handles_labels()[1] + ['Daylight-saving change'])
ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "clusters_by_month_2014_2026")
