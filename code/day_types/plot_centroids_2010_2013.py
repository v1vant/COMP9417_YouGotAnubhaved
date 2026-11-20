"""Fixed 2010-2013 day-type centroids (team k-means, k=2), shown as demand / daily mean by half-hour.
Run from the repo root: python code/day_types/plot_centroids_2010_2013.py"""
import numpy as np
from common import team_day_types, pyplot, save, COLOURS

types, _, typical, _ = team_day_types("2013-12-31")
counts = np.bincount(types["cluster"].to_numpy())
plt = pyplot()
fig, ax = plt.subplots(figsize=(7.5, 4.6)); hours = np.arange(48) / 2
for k in range(2):
    ax.plot(hours, typical[k], color=COLOURS[k], lw=2.5, label=f'Type {k} · {counts[k]:,} days (2010–2013)')
ax.axhline(1, color='#98A2B3', lw=.8, zorder=0)
ax.set(title='Day-type centroids, fitted on 2010–2013', xlabel='Interval start (AEST)', ylabel='Demand / daily mean',
       xlim=(0, 24), ylim=(.68, 1.26))
ax.set_xticks([0, 6, 12, 18, 24], ['00:00', '06:00', '12:00', '18:00', '24:00'])
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "day_type_centroids_2010_2013")
