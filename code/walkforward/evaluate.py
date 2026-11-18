"""Score every completed configuration and blend (pure test; not part of the repo).
Thresholds, blend weights and stackers are chosen on 2014 only. Configurations are compared on
2015-2019 (selection half); 2020-2024 (confirmation half) is reported but drives no choice."""
import itertools, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas, log_loss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
E = harness.E
RUNS = harness.RUNS
GRID = np.array([0.5]) if os.environ.get("FIXED_THRESHOLD") else np.round(np.linspace(0.2, 0.8, 61), 3)
SEL, CONF, ALL = list(range(2015, 2020)), list(range(2020, 2025)), list(range(2015, 2025))
truth_df = pd.read_parquet(f"{E}/truth.parquet")
from model import market_days
days = market_days(pd.DatetimeIndex(np.load(f"{E}/index.npy", allow_pickle=True)))
assert days.max() == pd.Timestamp("2024-12-31")
Y = {y: truth_df["price"].to_numpy()[np.asarray(days.year == y)].astype(bool) for y in range(2014, 2025)}
DAY = {y: pd.factorize(days[np.asarray(days.year == y)])[0] for y in range(2014, 2025)}
configs = json.load(open(f"{RUNS}/configs.json"))


def load_config(name):
    out, seconds = {}, {}
    for fit_end, years in harness.SCHEDULE:
        p = f"{RUNS}/{name}__{fit_end}.npz"
        if not os.path.exists(p):
            return None, None
        z = np.load(p)
        out.update({y: z[f"y{y}"] for y in years})
        seconds[fit_end] = json.load(open(p + ".json"))["seconds"]
    return out, seconds


def tune(p14):
    scores = [bas(Y[2014], p14 > t) for t in GRID]
    return float(GRID[int(np.argmax(scores))])


def score(P, thr):
    row = {"threshold": thr}
    for label, years in (("sel", SEL), ("conf", CONF), ("all", ALL)):
        t = np.concatenate([Y[y] for y in years]); p = np.concatenate([P[y] for y in years])
        row[f"BA_{label}"] = 100 * bas(t, p > thr)
        row[f"acc_{label}"] = 100 * np.mean((p > thr) == t)
    t = np.concatenate([Y[y] for y in ALL]); p = np.concatenate([P[y] for y in ALL])
    row["logloss_all"] = log_loss(t, np.clip(p, 1e-6, 1 - 1e-6))
    row["BA_2014"] = 100 * bas(Y[2014], P[2014] > thr)
    for y in ALL:
        row[f"BA_{y}"] = 100 * bas(Y[y], P[y] > thr)
    return row


def main():
    probs, rows = {}, []
    for name in configs:
        P, sec = load_config(name)
        if P is None:
            continue
        probs[name] = P
        r = {"model": name, **score(P, tune(P[2014])),
             "fit_min_14yr": sec[2023] / 60, "fit_min_full_2010_2024_est": sec[2023] / 60 * 15 / 14}
        r.update({f"{k}_at_0.5": v for k, v in score(P, 0.5).items() if k in ("BA_sel", "BA_conf", "BA_all", "acc_all")})
        rows.append(r)
    table = pd.DataFrame(rows).sort_values("BA_sel", ascending=False)
    table.to_csv(f"{E}/results_single{'_t05' if os.environ.get('FIXED_THRESHOLD') else ''}.csv", index=False)
    cols = ["model", "threshold", "BA_2014", "BA_sel", "BA_conf", "BA_all", "acc_all", "logloss_all", "BA_all_at_0.5", "fit_min_full_2010_2024_est"]
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(f"SINGLE MODELS ({len(table)} complete of {len(configs)}), ranked by 2015-2019 BA (threshold tuned on 2014):")
        print(table[cols].round(3).to_string(index=False))
    if "lgbm31" in probs:
        t = np.concatenate([Y[y] for y in ALL]); p = np.concatenate([probs["lgbm31"][y] for y in ALL])
        print(f"\nReference check, LightGBM-31 at Anubhav's 0.57: accuracy {100*np.mean((p>0.57)==t):.2f}, BA {100*bas(t, p>0.57):.2f} "
              f"(his feature_comparison_2015_2024: 80.60 / 79.53)")
    slow = table[table.fit_min_full_2010_2024_est > 120]
    if len(slow):
        print("\nOver the 2-hour limit (full-window estimate):", list(slow.model))

    # ---------- blends ----------
    lg = [m for m in table.model if m.startswith("lgbm") and "_D5" not in m]
    nn = [m for m in table.model if (m.startswith("nn") or m.startswith("torch")) and "_D5" not in m]
    short_lg, short_nn = lg[:3], nn[:3]
    for ref in ("lgbm63", "nn256x256"):
        if ref in probs and ref not in short_lg + short_nn:
            (short_lg if ref.startswith("lgbm") else short_nn).append(ref)
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    sig = lambda z: 1 / (1 + np.exp(-z))
    brows = []
    def add(name, P, chosen):
        brows.append({"blend": name, "chosen_on_2014": chosen, **score(P, tune(P[2014]))})
    W = np.round(np.linspace(0, 1, 21), 2)
    for a, b in itertools.product(short_lg, short_nn):
        for kind in ("prob", "logit"):
            mix = (lambda w: {y: w * probs[a][y] + (1 - w) * probs[b][y] for y in probs[a]}) if kind == "prob" else \
                  (lambda w: {y: sig(w * logit(probs[a][y]) + (1 - w) * logit(probs[b][y])) for y in probs[a]})
            best = max(W, key=lambda w: max(bas(Y[2014], mix(w)[2014] > t) for t in GRID))
            add(f"{kind}: {best:.2f} {a} + {1-best:.2f} {b}", mix(best), "weight+threshold")
        add(f"prob: 0.70 {a} + 0.30 {b} (fixed)", {y: 0.7 * probs[a][y] + 0.3 * probs[b][y] for y in probs[a]}, "threshold")
    if len(short_lg) >= 2 and short_nn:
        a, a2, b = short_lg[0], short_lg[1], short_nn[0]
        best, bw = -1, None
        for w1 in np.round(np.arange(0, 1.01, 0.1), 1):
            for w2 in np.round(np.arange(0, 1.01 - w1, 0.1), 1):
                P14 = w1 * probs[a][2014] + w2 * probs[a2][2014] + (1 - w1 - w2) * probs[b][2014]
                s = max(bas(Y[2014], P14 > t) for t in GRID)
                if s > best:
                    best, bw = s, (w1, w2)
        w1, w2 = bw
        add(f"3-way: {w1:.1f} {a} + {w2:.1f} {a2} + {1-w1-w2:.1f} {b}",
            {y: w1 * probs[a][y] + w2 * probs[a2][y] + (1 - w1 - w2) * probs[b][y] for y in probs[a]}, "weights+threshold")
    stack_set = short_lg + short_nn
    if len(stack_set) >= 2:
        F = lambda y: np.column_stack([logit(probs[m][y]) for m in stack_set])
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(F(2014), Y[2014])
        add(f"stacking (logistic regression on 2014) over {stack_set}", {y: lr.predict_proba(F(y))[:, 1] for y in probs[stack_set[0]]}, "stacker+threshold")
    # ---------- kNN: diversity against the strongest models, and whether it helps the ensemble ----------
    kn = [m for m in table.model if m.startswith("knn")]
    if kn and lg and nn:
        best_lg, best_nn = lg[0], nn[0]
        thr = dict(zip(table.model, table.threshold))
        def wrong(m):
            return np.concatenate([(probs[m][y] > thr[m]) != Y[y] for y in ALL])
        drows = []
        for m in [best_nn] + kn[:5]:
            ea, eb = wrong(best_lg), wrong(m)
            n11, n00, n10, n01 = np.mean(~ea & ~eb), np.mean(ea & eb), np.mean(~ea & eb), np.mean(ea & ~eb)
            drows.append({"model": m, "vs": best_lg, "Yule_Q": (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01),
                          "error_corr": np.corrcoef(ea, eb)[0, 1], "disagree_%": 100 * (n10 + n01),
                          "right_when_disagree_%": 100 * n01 / (n10 + n01)})
        with pd.option_context("display.width", 250):
            print("\nDIVERSITY vs the best LightGBM (pooled 2015-2024, each at its 2014 threshold):")
            print(pd.DataFrame(drows).round(3).to_string(index=False))
        base2 = None
        best = max(W, key=lambda w: max(bas(Y[2014], w * probs[best_lg][2014] + (1 - w) * probs[best_nn][2014] > t) for t in GRID))
        base2 = {y: best * probs[best_lg][y] + (1 - best) * probs[best_nn][y] for y in probs[best_lg]}
        add(f"REFERENCE 2-way: {best:.2f} {best_lg} + {1-best:.2f} {best_nn}", base2, "weight+threshold")
        for m in kn[:3]:
            for partner in (best_lg, best_nn):
                wb = max(W, key=lambda w: max(bas(Y[2014], w * probs[partner][2014] + (1 - w) * probs[m][2014] > t) for t in GRID))
                add(f"kNN pair: {wb:.2f} {partner} + {1-wb:.2f} {m}", {y: wb * probs[partner][y] + (1 - wb) * probs[m][y] for y in probs[m]}, "weight+threshold")
            bestv, bw = -1, None
            for w1 in np.round(np.arange(0, 1.001, 0.05), 2):
                for w2 in np.round(np.arange(0, 1.001 - w1, 0.05), 2):
                    P14 = w1 * probs[best_lg][2014] + w2 * probs[best_nn][2014] + (1 - w1 - w2) * probs[m][2014]
                    sc = max(bas(Y[2014], P14 > t) for t in GRID)
                    if sc > bestv:
                        bestv, bw = sc, (w1, w2)
            w1, w2 = bw
            add(f"3-way + kNN: {w1:.2f} {best_lg} + {w2:.2f} {best_nn} + {1-w1-w2:.2f} {m}",
                {y: w1 * probs[best_lg][y] + w2 * probs[best_nn][y] + (1 - w1 - w2) * probs[m][y] for y in probs[m]}, "weights+threshold")
        sset = [best_lg, best_nn] + kn[:2]
        F2 = lambda y: np.column_stack([logit(probs[m][y]) for m in sset])
        lr2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(F2(2014), Y[2014])
        add(f"stacking with kNN over {sset}", {y: lr2.predict_proba(F2(y))[:, 1] for y in probs[best_lg]}, "stacker+threshold")
    btable = pd.DataFrame(brows).sort_values("BA_sel", ascending=False)
    btable.to_csv(f"{E}/results_blends{'_t05' if os.environ.get('FIXED_THRESHOLD') else ''}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 100):
        print("\nBLENDS, ranked by 2015-2019 BA:")
        print(btable[["blend", "chosen_on_2014", "threshold", "BA_2014", "BA_sel", "BA_conf", "BA_all", "acc_all", "logloss_all"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
