"""Walk-forward only: ADDITIVE contribution of each hour group to the period's BA gain from demand features
(contributions sum exactly to the all-hours gain; within-group BA is not additive because up-rates differ by hour)."""
import json, os, sys
os.environ["FIXED_THRESHOLD"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import evaluate as ev, harness

X, truth, days = harness.load()
assert np.array_equal(np.asarray(days), np.asarray(ev.days)), "row order differs from evaluate"
start_hour = (X.index - pd.Timedelta(minutes=30)).hour
G = {"overnight 00-06": range(0, 6), "morning 06-10": range(6, 10), "solar 10-16": range(10, 16),
     "evening 16-21": range(16, 21), "late 21-24": range(21, 24)}
grp = np.full(len(X), "", dtype=object)
for k, hrs in G.items():
    grp[np.isin(start_hour, list(hrs))] = k
yr = np.asarray(days.year); dnum = np.asarray(days.factorize()[0]); lab = truth["price"].to_numpy().astype(bool)
S5 = (9417, 9418, 9419, 9420, 9421); L = lambda n: ev.load_config(n)[0]
mean = lambda ns: {y: np.mean([L(n)[y] for n in ns], 0) for y in ev.ALL}
full = lambda p: np.concatenate([p[y] for y in ev.ALL])          # rows of 2015-2024 in table order
m15 = np.isin(yr, ev.ALL)
PAIRS = {"Suhan+A5 (B) - Suhan [also drops 2010]": (mean([f"suhan_c2ws_A5_s{s}" for s in S5]), mean([f"suhan_c2ws_s{s}" for s in S5])),
         "LGBM-63+A5 - LGBM-63 (both from 2011)": (L("lgbm63_A5_blend"), L("lgbm63_from2011")),
         "LGBM-63+D4 - LGBM-63 (both from 2011)": (L("lgbm63_D4_blend"), L("lgbm63_from2011"))}
PERIODS = {"2015-17": [2015, 2016, 2017], "2018-20": [2018, 2019, 2020], "2021-23": [2021, 2022, 2023], "2024": [2024]}
t15, g15, y15, d15 = lab[m15], grp[m15], yr[m15], dnum[m15]
print("up-rate by hour group (2015-24):", {k: round(100 * t15[g15 == k].mean(), 1) for k in G})
rng = np.random.default_rng(9417)
for title, (pa, pb) in PAIRS.items():
    A, B = full(pa) > .5, full(pb) > .5
    print(f"\n== {title}: additive contribution to the period's BA gain (points of BA; 95% CI)")
    rows = []
    for k in list(G) + ["ALL HOURS"]:
        r = {"hours": k}
        for pname, yrs in PERIODS.items():
            m = np.isin(y15, yrs); t = t15[m]; g = (g15[m] == k) if k != "ALL HOURS" else np.ones(m.sum(), bool)
            ids, inv = np.unique(d15[m], return_inverse=True)
            s = lambda x: np.bincount(inv, weights=x.astype(float), minlength=len(ids))
            a, b = A[m], B[m]
            dtp = g * ((a & t).astype(float) - (b & t).astype(float))
            dtn = g * ((~a & ~t).astype(float) - (~b & ~t).astype(float))
            c = np.stack([s(t), s(~t), s(dtp), s(dtn)], 1)
            f = lambda v: 50 * (v[:, 2] / v[:, 0] + v[:, 3] / v[:, 1])
            pt = f(c.sum(0)[None])[0]
            bs = f(np.stack([c[rng.integers(0, len(c), len(c))].sum(0) for _ in range(2000)]))
            r[pname] = f"{pt:+.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]"
        rows.append(r)
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(rows).set_index("hours").to_string())
    for pname, yrs in PERIODS.items():
        m = np.isin(y15, yrs); from sklearn.metrics import balanced_accuracy_score as bas
        print(f"   check {pname}: direct all-hours BA gain {100*(bas(t15[m], A[m]) - bas(t15[m], B[m])):+.3f}")
