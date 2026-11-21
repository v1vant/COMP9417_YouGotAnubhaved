"""Input-based drift detector of the team code (analysis_tools.OnlineDetector on day_type_analysis's input_drift_score):
daily score = distance of the day's mean-normalised 48-slot demand shape to the nearest fixed 2010-2013 k-means centroid
(PCA distance and reconstruction error combined); statistic = 28-day rolling mean; threshold = 99th percentile of the 2014
rolling means; monitoring from 2015 (as run_experiments.py does), 28-day cooldown between signals.
Descriptive figure through 2026-10-31 (2025-26 included at the user's request). Run from the repo root."""
import os
import numpy as np, pandas as pd
from common import team_day_types, team, pyplot, save, OUT, COLOURS, END

types, _, _, _ = team_day_types(END)
score = types["input_drift_score"]
ref = score[score.index.year == 2014].to_numpy()
det = team.OnlineDetector(ref)
stat, signals = {}, []
for day, value in score[score.index >= "2015-01-01"].items():
    if det.observe(day, value):
        signals.append(day)
    stat[day] = det.statistic
stat = pd.Series(stat)
roll_all = score.rolling(28).mean()
out = pd.DataFrame({"input_drift_score": score, "rolling_28d": roll_all, "detector_statistic": stat})
out["signal"] = out.index.isin(signals)
os.makedirs(OUT, exist_ok=True); out.to_csv(os.path.join(OUT, "input_drift_2010_2026.csv"), index_label="day")
print(f"threshold (99th pct of 2014 28-day means) = {det.threshold:.4f}")
print(f"{len(signals)} signals; first: {[d.date().isoformat() for d in signals[:5]]}; last: {signals[-1].date() if signals else None}")
print("signals per year:", pd.Series([d.year for d in signals]).value_counts().sort_index().to_dict())
first_cross = stat[stat > det.threshold].index.min()
print("first day the statistic exceeds the threshold:", first_cross.date() if pd.notna(first_cross) else None)
print("yearly median score:", score.groupby(score.index.year).median().round(3).to_dict())

plt = pyplot()
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.axvspan(pd.Timestamp("2010-01-01"), pd.Timestamp("2014-01-01"), color='#F2F4F7', zorder=0)
ax.axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2015-01-01"), color='#E4E7EC', zorder=0)
ymax = float(score.quantile(.999)) * 1.08
ax.text(pd.Timestamp("2012-01-01"), ymax * .97, 'clusters fitted\n(2010–2013)', ha='center', va='top', fontsize=8, color='#475467')
ax.text(pd.Timestamp("2014-07-01"), ymax * .97, 'threshold\ncalibrated\n(2014)', ha='center', va='top', fontsize=8, color='#475467')
ax.plot(score.index, score.values, color='#98A2B3', lw=.5, alpha=.6, label='Daily score')
ax.plot(roll_all.index, roll_all.values, color=COLOURS[0], lw=2, label='28-day rolling mean (detector statistic)')
ax.axhline(det.threshold, color='#344054', lw=1.2, ls='--', label=f'Threshold = {det.threshold:.3f} (99th pct of 2014)')
if signals:
    ax.scatter(signals, [det.threshold] * len(signals), marker='v', s=36, color=COLOURS[1], zorder=5,
               edgecolor='white', linewidth=.8, label=f'Drift signal ({len(signals)}; ≥ 28 days apart)')
ax.set(title='Input-based drift detector: distance of each day’s demand shape to the 2010–2013 day types',
       xlabel='Market day (to October 2026)', ylabel='Distance to nearest centroid\n(demand / daily mean)',
       ylim=(0, ymax), xlim=(pd.Timestamp("2010-01-01"), pd.Timestamp("2026-11-01")))
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), fontsize=8.5, ncol=4)
ax.grid(axis='y', color='#EAECF0', lw=.7); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "input_drift_2010_2026")
