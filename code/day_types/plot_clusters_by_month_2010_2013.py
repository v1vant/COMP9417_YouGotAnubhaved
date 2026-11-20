"""Day type (team k-means, k=2, fitted on 2010-2013) of every 2010-2013 day, counted by month, with the NSW
daylight-saving changes marked (start: first Sunday of October; end: first Sunday of April). Run from the repo root."""
import os
import numpy as np, pandas as pd
from common import team_day_types, pyplot, save, OUT, COLOURS, NAMES

types, _, _, _ = team_day_types("2013-12-31")
t = types.reset_index(); t["month"] = t["day"].dt.to_period("M")
counts = pd.crosstab(t["month"], t["cluster"]).reindex(columns=[0, 1], fill_value=0)
os.makedirs(OUT, exist_ok=True); counts.to_csv(os.path.join(OUT, "clusters_by_month_2010_2013.csv"))
print(counts.T.to_string())


def first_sunday(y, m):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(6 - d.weekday()) % 7)


pos = lambda ts: (ts.year - 2010) * 12 + (ts.month - 1) + (ts.day - 1) / ts.days_in_month - 0.5
changes = [first_sunday(y, m) for y in range(2010, 2014) for m in (4, 10)]
months = counts.index; x = np.arange(len(months))
plt = pyplot()
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.bar(x, counts[0], width=.8, color=COLOURS[0], edgecolor='white', linewidth=1, label=NAMES[0])
ax.bar(x, counts[1], width=.8, bottom=counts[0], color=COLOURS[1], edgecolor='white', linewidth=1, label=NAMES[1])
for ts in changes:
    ax.axvline(pos(ts), color='#475467', lw=1, ls='--', zorder=0)
ax.text(pos(first_sunday(2010, 10)) + .15, 33.2, 'daylight saving\nstarts', fontsize=8, color='#475467', va='top')
ax.text(pos(first_sunday(2011, 4)) + .15, 33.2, 'daylight saving\nends', fontsize=8, color='#475467', va='top')
ax.set_xticks(x[::3], [m.strftime('%b\n%Y') if m.month == 1 else m.strftime('%b') for m in months[::3]])
ax.set(ylabel='Days in month', ylim=(0, 34), xlim=(-.7, len(months) - .3),
       title='Day type by month, 2010–2013 (clusters fitted on these years)')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, fontsize=9)
ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "clusters_by_month_2010_2013")
