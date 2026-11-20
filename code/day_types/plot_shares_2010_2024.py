"""The two figures of save_figures() in code/analysis_tools.py (same plotting code), but with day_type_analysis fed data
up to 2024-12-31 only; run_experiments.py feeds the full CSV, so its shares plot includes the 2025-26 holdout."""
import os
import numpy as np
from common import team_day_types, pyplot, OUT

types, shares, typical, _ = team_day_types("2024-12-31")
os.makedirs(OUT, exist_ok=True); shares.to_csv(os.path.join(OUT, "day_type_shares_2010_2024.csv"))
print(shares.round(3).to_string())
plt = pyplot()
plt.rcParams.update(plt.rcParamsDefault)                     # save_figures uses matplotlib defaults
fig, ax = plt.subplots(figsize=(9, 5))
for k, profile in enumerate(typical):
    ax.plot(np.arange(48) / 2, profile, label=f"Cluster {k}")
ax.set(xlabel="Interval start (AEST hours)", ylabel="Demand / daily mean (unitless)", title="Fixed training-period day types")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "day_type_profiles.png"), dpi=160); plt.close(fig)
ax = shares.plot.area(figsize=(10, 5))
ax.set(xlabel="Year", ylabel="Share of days (fraction)", title="Day-type prevalence (2010–2024)")
ax.figure.tight_layout(); ax.figure.savefig(os.path.join(OUT, "day_type_shares_2010_2024.png"), dpi=160); plt.close(ax.figure)
print("saved", os.path.join(OUT, "day_type_profiles.png"), "and day_type_shares_2010_2024.png")
