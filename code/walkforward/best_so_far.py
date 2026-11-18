"""Pure test (not in the repo): the current best-so-far configuration and a 'versus best' table for every new row.
Update BEST when a new configuration beats it (2015-2024 BA at 0.5). Note: the best-so-far is itself chosen by looking
at 2015-2024, so small margins over it are selection-biased."""
import numpy as np, pandas as pd
import evaluate as ev
from sklearn.metrics import balanced_accuracy_score as bas

BEST = {"label": "BEST SO FAR: 70/30 LGBM-63+A5 (his fc) + NN 256x256+D4 (his fc)",
        "parts": [("lgbm63_A5_blend", 0.7), ("nn256_D4_blend", 0.3)]}
YRS = list(range(2014, 2025))
cat = lambda d, yrs: np.concatenate([d[y] for y in yrs])


def best_probs():
    ps = [(ev.load_config(n)[0], w) for n, w in BEST["parts"]]
    return {y: sum(w * p[y] for p, w in ps) for y in YRS}


def _boot(pa, pb, yrs, n, seed=9417):
    t = cat(ev.Y, yrs); A = cat(pa, yrs) > .5; B = cat(pb, yrs) > .5
    day = np.concatenate([np.asarray(ev.DAY[y]) + 1000 * y for y in yrs]); ids, inv = np.unique(day, return_inverse=True)
    s = lambda x: np.bincount(inv, weights=x.astype(float), minlength=len(ids))
    c = np.stack([s(t), s(~t), s(A & t), s(~A & ~t), s(B & t), s(~B & ~t), s(A == t), s(B == t), s(np.ones(len(t)))], 1)
    m = lambda v: (50 * (v[:, 2] / v[:, 0] + v[:, 3] / v[:, 1]) - 50 * (v[:, 4] / v[:, 0] + v[:, 5] / v[:, 1]), 100 * (v[:, 6] - v[:, 7]) / v[:, 8])
    pt = m(c.sum(0)[None]); rng = np.random.default_rng(seed)
    g, a = m(np.stack([c[rng.integers(0, len(c), len(c))].sum(0) for _ in range(n)]))
    return pt[0][0], np.percentile(g, [2.5, 97.5]), pt[1][0], np.percentile(a, [2.5, 97.5])


def versus_best(rows, n=2000):
    """rows: {label: {year: probabilities}}. Returns one line per row: its BA and its gap to the best so far."""
    best = best_probs()
    out = []
    for k, p in rows.items():
        if p is None:
            continue
        g, ci, ga, cia = _boot(p, best, ev.ALL, n)
        g1 = _boot(p, best, ev.SEL, 1)[0]; g2 = _boot(p, best, ev.CONF, 1)[0]
        wins = sum(bas(ev.Y[y], p[y] > .5) > bas(ev.Y[y], best[y] > .5) for y in ev.ALL)
        out.append({"model": k, "BA_15-24": 100 * bas(cat(ev.Y, ev.ALL), cat(p, ev.ALL) > .5),
                    "vs_best_BA": round(g, 3), "CI_low": round(ci[0], 2), "CI_high": round(ci[1], 2),
                    "vs_best_acc": round(ga, 3), "acc_CI": f"[{cia[0]:+.2f},{cia[1]:+.2f}]", "years_won": wins,
                    "vs_best_15-19": round(g1, 2), "vs_best_20-24": round(g2, 2)})
    T = pd.DataFrame(out).set_index("model").sort_values("vs_best_BA", ascending=False)
    return T


def report(rows, title="vs BEST SO FAR"):
    best = best_probs()
    b = 100 * bas(cat(ev.Y, ev.ALL), cat(best, ev.ALL) > .5)
    acc = 100 * np.mean((cat(best, ev.ALL) > .5) == cat(ev.Y, ev.ALL))
    print(f"\n== {title}: {BEST['label']} = {b:.3f} BA / {acc:.3f}% accuracy (2015-2024). Rows sorted by gap; "
          f"CI = 95% paired day-block bootstrap; years_won of 10.")
    with pd.option_context("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 200):
        print(versus_best(rows).round(3).to_string())
