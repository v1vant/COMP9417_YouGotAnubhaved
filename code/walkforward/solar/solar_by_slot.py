"""Walk-forward only (2015-2024): does the gain from demand-forecast features shrink mainly in solar (midday) hours?"""
import json, os, sys
os.environ["FIXED_THRESHOLD"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import evaluate as ev, harness
from sklearn.metrics import balanced_accuracy_score as bas

X, truth, days = harness.load()
C = json.load(open(f"{harness.RUNS}/configs.json"))
for n in ("lgbm63_from2011", "lgbm63_A5_blend", "lgbm63_D4_blend"):
    print(n, json.dumps(C[n]))
start_hour = (X.index - pd.Timedelta(minutes=30)).hour          # timestamps mark the END of each half-hour
G = {"overnight 00-06": range(0, 6), "morning 06-10": range(6, 10), "solar 10-16": range(10, 16),
     "evening 16-21": range(16, 21), "late 21-24": range(21, 24)}
grp = np.full(len(X), "", dtype=object)
for k, hrs in G.items():
    grp[np.isin(start_hour, list(hrs))] = k
yr = np.asarray(days.year)
S5 = (9417, 9418, 9419, 9420, 9421)
L = lambda n: ev.load_config(n)[0]
mean = lambda ns: {y: np.mean([L(n)[y] for n in ns], 0) for y in ev.ALL}
PAIRS = {
    "Suhan+A5 (B) minus Suhan [also drops 2010]": (mean([f"suhan_c2ws_A5_s{s}" for s in S5]), mean([f"suhan_c2ws_s{s}" for s in S5])),
    "LGBM-63+A5 minus LGBM-63 (both from 2011)": (L("lgbm63_A5_blend"), L("lgbm63_from2011")),
    "LGBM-63+D4 minus LGBM-63 (both from 2011)": (L("lgbm63_D4_blend"), L("lgbm63_from2011")),
}
PERIODS = {"2015-17": [2015, 2016, 2017], "2018-20": [2018, 2019, 2020], "2021-23": [2021, 2022, 2023], "2024": [2024]}


def boot(a, b, t, d, n=2000):
    """paired day-block bootstrap of BA(a) - BA(b)"""
    ids, inv = np.unique(d, return_inverse=True)
    s = lambda x: np.bincount(inv, weights=x.astype(float), minlength=len(ids))
    A, B = a > .5, b > .5
    c = np.stack([s(t), s(~t), s(A & t), s(~A & ~t), s(B & t), s(~B & ~t)], 1)
    f = lambda v: 50 * (v[:, 2] / v[:, 0] + v[:, 3] / v[:, 1]) - 50 * (v[:, 4] / v[:, 0] + v[:, 5] / v[:, 1])
    rng = np.random.default_rng(9417)
    g = f(np.stack([c[rng.integers(0, len(c), len(c))].sum(0) for _ in range(n)]))
    return f(c.sum(0)[None])[0], np.percentile(g, [2.5, 97.5])


dnum = np.asarray(days.factorize()[0])
for title, (pa, pb) in PAIRS.items():
    print(f"\n== {title}: BA gain from the demand features, by hour group and period (95% CI)")
    rows = []
    for gname in list(G) + ["ALL HOURS"]:
        r = {"hours": gname}
        for pname, yrs in PERIODS.items():
            m = np.isin(yr, yrs) & ((grp == gname) if gname != "ALL HOURS" else True)
            t = truth["price"].to_numpy()[m].astype(bool)
            # walk-forward predictions are stored per year in row order; rebuild the row-aligned arrays
            a = np.concatenate([pa[y][(grp[yr == y] == gname) if gname != "ALL HOURS" else slice(None)] for y in yrs])
            b = np.concatenate([pb[y][(grp[yr == y] == gname) if gname != "ALL HOURS" else slice(None)] for y in yrs])
            pt, ci = boot(a, b, t, dnum[m])
            r[pname] = f"{pt:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]"
        rows.append(r)
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(rows).set_index("hours").to_string())

# context: how much did the share of 'up' labels and the demand-price link change at midday?
print("\n== Midday (10-16) context by year: mean actual demand (MW), corr(next-day demand change vs label) per hour group")
d_act = truth["demand"].to_numpy(); prevmean = X["nsw_demand_previous_mean"].to_numpy(); lab = truth["price"].to_numpy()
for y in ev.ALL:
    m = yr == y
    out = []
    for gname in ("solar 10-16", "evening 16-21"):
        mm = m & (grp == gname)
        out.append(f"{gname}: demand {d_act[mm].mean():6.0f}, corr(demand - prev-day mean, up) {np.corrcoef(d_act[mm] - prevmean[mm], lab[mm])[0,1]:+.3f}")
    print(y, " | ".join(out))
