"""Fix (a), pre-declared: use demand-feature members only outside 10:00-16:00. In solar hours each demand member is
replaced by its demand-free twin (A_s -> lgbm63ws_from2011_s, B_s -> suhan_c2ws_from2011_s, nn256_D4_blend ->
nn256_from2011); nn64 is unchanged. Walk-forward 2015-2024 only, threshold 0.5.
Works if (W1) it beats F = 0.7*Suhan(5 seeds) + 0.3*NN64 over 2015-24 with CI lower bound > 0, and
(W2) its gain over F is >= 0 in 2024 and in 2022-24 (point estimates)."""
import os, sys
os.environ["FIXED_THRESHOLD"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import evaluate as ev, best_so_far as bsf, harness
from sklearn.metrics import balanced_accuracy_score as bas

S5 = (9417, 9418, 9419, 9420, 9421); YRS = list(range(2014, 2025)); L = lambda n: ev.load_config(n)[0]
X, truth, days = harness.load()
assert np.array_equal(np.asarray(days), np.asarray(ev.days))
hour = (X.index - pd.Timedelta(minutes=30)).hour; yr = np.asarray(days.year)
SOLAR = {y: np.isin(hour[yr == y], range(10, 16)) for y in YRS}
TWIN = {**{f"lgbm63_D4ws_s{s}": f"lgbm63ws_from2011_s{s}" for s in S5},
        **{f"suhan_c2ws_A5_s{s}": f"suhan_c2ws_from2011_s{s}" for s in S5}, "nn256_D4_blend": "nn256_from2011"}
missing = [t for t in TWIN.values() if L(t) is None]
if missing:
    sys.exit(f"twins not finished: {missing}")


def blend(parts, gate):
    """gate: 'none' (champion), 'solar' (twins in 10-16 only), 'all' (twins everywhere = demand-free champion)."""
    out = {}
    for y in YRS:
        p = np.zeros(len(ev.Y[y]))
        for n, w in parts:
            a = L(n)[y]
            if n in TWIN and gate != "none":
                t = L(TWIN[n])[y]
                a = np.where(SOLAR[y], t, a) if gate == "solar" else t
            p += w * a
        out[y] = p
    return out


CH = blend(bsf.BEST["parts"], "none"); GA = blend(bsf.BEST["parts"], "solar"); DF = blend(bsf.BEST["parts"], "all")
SU = {y: np.mean([L(f"suhan_c2ws_s{s}")[y] for s in S5], 0) for y in YRS}
F = {y: 0.7 * SU[y] + 0.3 * L("nn64")[y] for y in YRS}
ba = lambda p, ys: 100 * bas(np.concatenate([ev.Y[y] for y in ys]), np.concatenate([p[y] for y in ys]) > .5)
P = {"(a) gated champion [PRIMARY]": GA, "champion (best so far)": CH, "F = 0.7 Suhan + 0.3 NN64 (fallback)": F,
     "Suhan alone": SU, "[secondary] demand-free champion (twins at all hours)": DF}
print(f"{'model':55s} {'2014':>7s} {'15-24':>7s} {'15-19':>7s} {'20-24':>7s} {'22-24':>7s} {'2024':>7s}")
for k, p in P.items():
    print(f"{k:55s} {ba(p,[2014]):7.3f} {ba(p,ev.ALL):7.3f} {ba(p,ev.SEL):7.3f} {ba(p,ev.CONF):7.3f} {ba(p,[2022,2023,2024]):7.3f} {ba(p,[2024]):7.3f}")
print("\nPaired day-block bootstrap (4000 draws): row minus reference")
for k in ("(a) gated champion [PRIMARY]", "champion (best so far)", "[secondary] demand-free champion (twins at all hours)"):
    for ref in ("F = 0.7 Suhan + 0.3 NN64 (fallback)", "champion (best so far)", "Suhan alone"):
        if k == ref:
            continue
        g, ci, ga, _ = bsf._boot(P[k], P[ref], ev.ALL, 4000)
        g22 = bsf._boot(P[k], P[ref], [2022, 2023, 2024], 4000); g24 = bsf._boot(P[k], P[ref], [2024], 4000)
        print(f"  {k[:40]:40s} - {ref[:36]:36s} 15-24 {g:+.3f} [{ci[0]:+.2f},{ci[1]:+.2f}] acc {ga:+.3f} | "
              f"22-24 {g22[0]:+.3f} [{g22[1][0]:+.2f},{g22[1][1]:+.2f}] | 2024 {g24[0]:+.3f} [{g24[1][0]:+.2f},{g24[1][1]:+.2f}]")
print("\nPer-year BA minus F:")
for k in ("(a) gated champion [PRIMARY]", "champion (best so far)", "[secondary] demand-free champion (twins at all hours)"):
    print(f"  {k[:40]:40s} " + " ".join(f"{y}:{ba(P[k],[y]) - ba(F,[y]):+.2f}" for y in YRS))
g, ci, *_ = bsf._boot(GA, F, ev.ALL, 4000)
w1 = ci[0] > 0; w2 = (ba(GA, [2024]) - ba(F, [2024]) >= 0) and (ba(GA, [2022, 2023, 2024]) - ba(F, [2022, 2023, 2024]) >= 0)
print(f"\nFIX (a) VERDICT: W1 (beats F 2015-24, CI low > 0) {'PASS' if w1 else 'FAIL'}; W2 (>= F in 2024 and 2022-24) "
      f"{'PASS' if w2 else 'FAIL'} -> {'WORKS' if (w1 and w2) else 'DOES NOT WORK'}")
