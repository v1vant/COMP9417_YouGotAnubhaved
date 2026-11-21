"""Distance of each day to the centroid of its assigned day type, by year 2014-2026 (2026 = Jan-Oct): min / median / max per type.
Distance = the 'distance' column of day_type_analysis (Euclidean distance in PCA space to the nearest = assigned
k-means centroid; PCA and k-means fitted on 2010-2013). Run from the repo root."""
import os
from common import team_day_types, pyplot, save, OUT, COLOURS, NAMES, END

types, _, _, _ = team_day_types(END)
t = types.reset_index()
ref = t[t.year <= 2013].groupby("cluster")["distance"].median()
stats = t[t.year >= 2014].groupby(["cluster", "year"])["distance"].agg(["min", "median", "max", "size"]).reset_index()
os.makedirs(OUT, exist_ok=True); stats.to_csv(os.path.join(OUT, "centroid_distance_by_year.csv"), index=False)
print(stats.round(4).to_string(index=False)); print("2010-2013 median by type:", ref.round(4).to_dict())
plt = pyplot()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
ymax = stats["max"].max() * 1.12
for k, ax in enumerate(axes):
    s = stats[stats.cluster == k]
    ax.fill_between(s.year, s["min"], s["max"], color=COLOURS[k], alpha=.16, lw=0, label='Range (min–max)')
    ax.plot(s.year, s["max"], color=COLOURS[k], lw=1, alpha=.55)
    ax.plot(s.year, s["min"], color=COLOURS[k], lw=1, alpha=.55)
    ax.plot(s.year, s["median"], color=COLOURS[k], lw=2.2, marker='o', ms=4, label='Median')
    ax.axhline(ref[k], color='#667085', lw=1, ls=':', label='2010–2013 median')
    for _, r in s.iterrows():
        ax.text(r.year, ymax * .965, f"{int(r['size'])}", ha='center', va='top', fontsize=7.5, color='#667085')
    ax.text(2013.6, ymax * .965, 'days:', ha='right', va='top', fontsize=7.5, color='#667085')
    ax.set(title=NAMES[k], xlabel='Year', xlim=(2013.5, 2026.5), ylim=(0, ymax))
    ax.set_xticks(range(2014, 2027, 2))
    ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
axes[0].set_ylabel('Distance to assigned centroid\n(Euclidean, PCA space)')
axes[1].legend(loc='center right', fontsize=8.5)
fig.suptitle('Distance of each day to its day-type centroid (clusters fixed on 2010–2013; 2026 = Jan–Oct)', fontsize=13, fontweight='bold')
fig.tight_layout(); save(fig, "centroid_distance_by_year")
