"""Fix (b) pre-declared gate: the solar-aware demand forecast must cut midday (10:00-16:00) MAPE over 2021-2024 by >= 10%
relative to the control rebuild (same code, no new inputs), without raising all-hours MAPE over 2015-2024."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, harness
X, truth, days = harness.load()
d = truth["demand"].to_numpy(float); yr = np.asarray(days.year)
h = (X.index - pd.Timedelta(minutes=30)).hour
G = {"overnight": range(0, 6), "morning": range(6, 10), "MIDDAY 10-16": range(10, 16), "evening": range(16, 21), "late": range(21, 24)}
F = {k: np.load(f"{harness.RUNS}/{k}_blend_forecast.npy") for k in ("anubhav", "ctrl", "solar")}
mape = lambda f, m: 100 * np.mean(np.abs(f[m] - d[m]) / d[m])
rows = []
for y in range(2011, 2025):
    r = {"year": y}
    for k in F:
        r[f"{k} all"] = mape(F[k], yr == y)
    for k in ("ctrl", "solar"):
        r[f"{k} midday"] = mape(F[k], (yr == y) & np.isin(h, list(G["MIDDAY 10-16"])))
    rows.append(r)
T = pd.DataFrame(rows).set_index("year")
print("Demand forecast MAPE (%), out-of-sample, by year"); print(T.round(2).to_string())
mid = np.isin(h, list(G["MIDDAY 10-16"])); r21 = np.isin(yr, range(2021, 2025)); r15 = np.isin(yr, range(2015, 2025))
print("\nBy hour group, 2021-2024 pooled (ctrl -> solar):")
for k, hrs in G.items():
    m = r21 & np.isin(h, list(hrs))
    print(f"  {k:13s} {mape(F['ctrl'], m):.3f} -> {mape(F['solar'], m):.3f}  ({100*(mape(F['solar'], m)/mape(F['ctrl'], m)-1):+.1f}%)")
cm, sm = mape(F["ctrl"], r21 & mid), mape(F["solar"], r21 & mid)
ca, sa = mape(F["ctrl"], r15), mape(F["solar"], r15)
print(f"\nControl vs Anubhav's saved stream 2015-2024 all-hours MAPE: {ca:.3f} vs {mape(F['anubhav'], r15):.3f}; "
      f"max |diff| 2015-24 {np.nanmax(np.abs(F['ctrl'][r15] - F['anubhav'][r15])):.1f} MW")
# day-block bootstrap of the relative midday change 2021-24
dd = np.asarray(days.factorize()[0])
m = r21 & mid; ids, inv = np.unique(dd[m], return_inverse=True)
ec = np.bincount(inv, np.abs(F["ctrl"][m] - d[m]) / d[m]); es = np.bincount(inv, np.abs(F["solar"][m] - d[m]) / d[m])
rng = np.random.default_rng(9417); bs = []
for _ in range(2000):
    i = rng.integers(0, len(ids), len(ids)); bs.append(100 * (es[i].sum() / ec[i].sum() - 1))
print(f"GATE 1: midday 2021-24 MAPE ctrl {cm:.3f} -> solar {sm:.3f} = {100*(sm/cm-1):+.1f}% [{np.percentile(bs,2.5):+.1f}, {np.percentile(bs,97.5):+.1f}] "
      f"(needs <= -10%): {'PASS' if sm <= 0.9 * cm else 'FAIL'}")
print(f"GATE 2: all-hours 2015-24 MAPE ctrl {ca:.3f} -> solar {sa:.3f} (must not rise): {'PASS' if sa <= ca else 'FAIL'}")
