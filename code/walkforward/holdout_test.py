# NOTE (review package): this is the exact script that produced results/walkforward/holdout/ on 2026-11-19. It ran from
# the scratch experiment directory, whose layout matches code/walkforward plus its data cache; it is kept for
# provenance and has NOT been re-run here (re-running would score the holdout again).
"""FINAL HOLDOUT TEST (run once, at the user's request on 2026-11-19).
Protocol (the user's earlier rule for this test): no training on 2025 or 2026. Every model -- the demand forecaster and
all price models -- is fitted on market days up to 2024-12-31 only, then predicts 2025-01-01 .. 2026-10-31 at the fixed
0.5 threshold. Nothing is tuned here: the champion and baselines are frozen exactly as chosen on 2014-2024.

Stages: prep (features, labels, demand forecast for 2025-26), fit (one job per model), score."""
import json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

E = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/thomas/Downloads/COMP9417_YouGotAnubhaved"
H = f"{E}/holdout"
CSV = f"{REPO}/data/nem_weather_halfhourly_2010_2026.csv"
S5 = (9417, 9418, 9419, 9420, 9421)


def market_days(index):
    return (pd.DatetimeIndex(index) - pd.Timedelta(minutes=30)).normalize()


# ---------------------------------------------------------------- stage 1: prep
def mlp_worker(src, dst):
    sys.path.insert(0, f"{REPO}/code")
    from demand_mlp import fit_demand_mlp, predict_demand_mlp
    xtr, ytr, xte = pd.read_pickle(src)
    bundle = fit_demand_mlp(xtr, ytr, epochs=10)
    np.save(dst, predict_demand_mlp(bundle, xte).astype(np.float64))


def prep():
    import harness
    from model import load_data, training_table
    from train import fit_model
    from model import predict_task
    os.makedirs(H, exist_ok=True)
    X, y = training_table(load_data(CSV), weather_shape=True)
    days = market_days(X.index)
    assert days.max() == pd.Timestamp("2026-10-31"), days.max()
    ws_names = json.load(open(f"{harness.RUNS}/weather_shape_names.json"))
    base_cols = json.load(open(f"{harness.E}/columns.json"))
    Xb = X[base_cols]
    # rows up to 2024 must be identical to the walk-forward cache
    Xh, th, dh = harness.load()
    m24 = np.asarray(days <= pd.Timestamp("2024-12-31"))
    assert Xb.index[m24].equals(Xh.index), "row sets differ"
    assert np.array_equal(Xb.to_numpy(np.float32)[m24], Xh.to_numpy(np.float32)), "features differ up to 2024"
    assert np.array_equal(y["price"].to_numpy()[m24], th["price"].to_numpy()), "labels differ up to 2024"
    print(f"full table {X.shape}; rows to 2024 identical to the walk-forward cache ({m24.sum()} rows); holdout rows {(~m24).sum()}", flush=True)
    # demand forecast: 2011-2024 = Anubhav's out-of-sample stream; 2025-26 = his recipe trained on days <= 2024
    f = np.full(len(X), np.nan)
    f[m24] = np.load(f"{harness.RUNS}/anubhav_blend_forecast.npy")
    columns = json.load(open(f"{REPO}/code/demand_feature_columns.json"))
    cfg = {"family": "boosted", "leaves": 63, "trees": 900, "learning_rate": 0.025}
    t0 = time.time()
    model = fit_model("demand", cfg, Xb[m24], y[m24])
    assert list(model["features"]) == columns, "demand LightGBM is not the 405-feature recipe"
    lg = predict_task(model, Xb[~m24])
    pd.to_pickle((Xb.loc[m24, columns], y.loc[m24, "demand"], Xb.loc[~m24, columns]), f"{H}/mlp_in.pkl")
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    subprocess.run([sys.executable, __file__, "mlp", f"{H}/mlp_in.pkl", f"{H}/mlp_out.npy"], check=True, env=env)
    os.remove(f"{H}/mlp_in.pkl")
    ml = np.load(f"{H}/mlp_out.npy")
    f[~m24] = 0.5 * (lg + ml)
    dem = y["demand"].to_numpy()[~m24]
    print(f"holdout demand forecast MAPE: LightGBM {100*np.mean(np.abs(lg-dem)/dem):.2f}%, MLP {100*np.mean(np.abs(ml-dem)/dem):.2f}%, "
          f"blend {100*np.mean(np.abs(f[~m24]-dem)/dem):.2f}% ({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{H}/X.npy", X[base_cols + ws_names].to_numpy(np.float32))
    np.save(f"{H}/index.npy", X.index.to_numpy())
    json.dump(base_cols + ws_names, open(f"{H}/columns.json", "w"))
    y.to_parquet(f"{H}/truth.parquet")
    np.save(f"{H}/anubhav_blend_forecast.npy", f)
    json.dump(ws_names, open(f"{H}/weather_shape_names.json", "w"))


def prep_neg():
    import build_negative_features as bnf
    raw = pd.read_csv(CSV, usecols=["timestamp", "nsw_price", "vic_price", "qld_price"], parse_dates=["timestamp"]).set_index("timestamp")
    idx = pd.DatetimeIndex(np.load(f"{H}/index.npy", allow_pickle=True))
    F = bnf.build(raw, idx)
    days = market_days(idx); m = np.asarray(days == days.max())
    raw2 = raw.copy(); raw2.loc[market_days(raw2.index) == days.max(), ["nsw_price", "vic_price", "qld_price"]] = -999.0
    assert np.allclose(F.to_numpy()[m], bnf.build(raw2, idx).to_numpy()[m], equal_nan=True), "look-ahead!"
    np.save(f"{H}/negprice.npy", F.to_numpy(np.float32)); json.dump(list(F.columns), open(f"{H}/negprice_names.json", "w"))
    d1 = pd.Series(F["nsw_neg_slots_D"].to_numpy(), index=np.asarray(days)).groupby(level=0).first()
    print("days after >= 6 NSW negative slots:", {int(y): int(((d1 >= 6) & (d1.index.year == y)).sum()) for y in (2023, 2024, 2025, 2026)})


# ---------------------------------------------------------------- stage 2: fits
def load_full():
    idx = pd.DatetimeIndex(np.load(f"{H}/index.npy", allow_pickle=True))
    cols = json.load(open(f"{H}/columns.json"))
    X = pd.DataFrame(np.load(f"{H}/X.npy", mmap_mode="r"), index=idx, columns=cols)
    return X, pd.read_parquet(f"{H}/truth.parquet"), market_days(idx)


def fit_one(job):
    name, spec = job
    out = f"{H}/pred_{name}.npy"
    if os.path.exists(out):
        return name
    import harness
    from train import fit_model
    from model import price_probability
    import lgbx
    X, y, days = load_full()
    ws = json.load(open(f"{H}/weather_shape_names.json"))
    Xb = X[[c for c in X.columns if c not in ws]].copy()
    if spec.get("demand"):
        harness.RUNS = H                          # add_demand_features reads RUNS/anubhav_blend_forecast.npy (full length)
        Xb = harness.add_demand_features(Xb, y, days, spec["demand"], False, "anubhav_blend")
    if spec.get("weather_shape"):
        for c in ws:
            Xb[c] = X[c].to_numpy()
    if spec.get("neg_features"):
        for k, c in enumerate(json.load(open(f"{H}/negprice_names.json"))):
            Xb[c] = np.load(f"{H}/negprice.npy", mmap_mode="r")[:, k]
    start = spec.get("start_year", 2010)
    train = np.asarray((days.year >= start) & (days <= pd.Timestamp("2024-12-31")))
    pred = np.asarray(days >= pd.Timestamp("2025-01-01"))
    assert not (train & pred).any() and days[train].max() <= pd.Timestamp("2024-12-31")
    t0 = time.time()
    if spec["kind"] == "nnx":
        import nn_ext
        params = nn_ext.fit(spec["config"], Xb[train], y["price"][train], z_clip=spec.get("z_clip"))
        prob = np.concatenate([nn_ext.predict(params, Xb[pred].iloc[i:i + 20000]) for i in range(0, int(pred.sum()), 20000)])
    elif spec["kind"] == "lgbx":
        z = np.zeros(int(train.sum()))
        prob, _ = lgbx.fit_predict(spec, Xb[train], Xb[pred], y["price"].to_numpy()[train], z, z)
    else:
        model = fit_model("price", spec["config"], Xb[train], y[train])
        prob = np.concatenate([price_probability(model, Xb[pred].iloc[i:i + 20000]) for i in range(0, int(pred.sum()), 20000)])
    np.save(out, prob.astype(np.float64))
    json.dump({"seconds": time.time() - t0, "train_rows": int(train.sum()), "train_end": str(days[train].max().date()),
               "pred_rows": int(pred.sum())}, open(out + ".json", "w"))
    return name


def jobs():
    C = json.load(open(f"{REPO}/results/walkforward/runs/configs.json"))
    J = [(n, C[n]) for n in [f"lgbm63_D4ws_s{s}" for s in S5] + [f"suhan_c2ws_A5_s{s}" for s in S5]
         + ["nn256_D4_blend", "nn64", "lgbm31", "lgbm63", "lgbm63_A5_blend"]]
    X = json.load(open(f"{E}/runs/extra_configs.json"))
    J.append(("nn256_D4neg_z3", X["nn256_D4neg_z3"][0]))          # secondary test: NN + D4 + negative features, cap +-3 sd
    J += [(f"suhan_c2ws_s{s}", X[f"suhan_c2ws_s{s}"][0]) for s in S5]  # BASELINE: Suhan's price-c2 package model (C2, 5 seeds)
    return J


# ---------------------------------------------------------------- stage 3: score
def score():
    """Every model at threshold 0.5 (the pre-declared rule) and at 0.57 (descriptive only: choosing a threshold from the
    holdout would be tuning on it)."""
    from sklearn.metrics import balanced_accuracy_score as bas
    X, y, days = load_full()
    pred = np.asarray(days >= pd.Timestamp("2025-01-01"))
    t = y["price"].to_numpy()[pred].astype(bool); d = np.asarray(days[pred]); yr = np.asarray(days[pred].year)
    P = {n: np.load(f"{H}/pred_{n}.npy") for n, _ in jobs() if os.path.exists(f"{H}/pred_{n}.npy")}
    have = lambda names: all(n in P for n in names)
    mean = lambda names: np.mean([P[n] for n in names], axis=0)
    LA, LB = [f"lgbm63_D4ws_s{s}" for s in S5], [f"suhan_c2ws_A5_s{s}" for s in S5]
    M = {}
    if have(LA + LB + ["nn256_D4_blend", "nn64"]):
        M["CHAMPION (final model)"] = 0.7 * (0.5 * mean(LA) + 0.5 * mean(LB)) + 0.3 * (0.5 * P["nn256_D4_blend"] + 0.5 * P["nn64"])
    if have([f"suhan_c2ws_s{s}" for s in S5]):
        M["BASELINE: Suhan's price-c2 model (C2, 5 seeds, weather shape, no demand)"] = mean([f"suhan_c2ws_s{s}" for s in S5])
    if have(["lgbm63_A5_blend", "nn256_D4_blend"]):
        M["previous best (0.7 LGBM-63+A5 + 0.3 NN+D4)"] = 0.7 * P["lgbm63_A5_blend"] + 0.3 * P["nn256_D4_blend"]
    if have(LA): M["LightGBM part A (LGBM-63+D4+ws, 5 seeds)"] = mean(LA)
    if have(LB): M["LightGBM part B (Suhan+A5+ws, 5 seeds)"] = mean(LB)
    for n, lab in (("nn256_D4_blend", "NN+D4"), ("nn64", "NN 64"), ("lgbm63", "LightGBM-63 (no demand)"),
                   ("lgbm31", "package LightGBM-31")):
        if n in P: M[lab] = P[n]
    if have(["nn256_D4neg_z3"]):
        M["[secondary] NN+D4+neg, inputs capped +-3 sd"] = P["nn256_D4neg_z3"]
        if have(LA + LB + ["nn64"]):
            M["[secondary] champion with NN+D4+neg(+-3) in place of NN+D4"] = 0.7 * (0.5 * mean(LA) + 0.5 * mean(LB)) + 0.3 * (0.5 * P["nn256_D4neg_z3"] + 0.5 * P["nn64"])
        if "lgbm63_A5_blend" in P:
            M["[secondary] previous best with NN+D4+neg(+-3)"] = 0.7 * P["lgbm63_A5_blend"] + 0.3 * P["nn256_D4neg_z3"]
    yday = (X["price_margin_1"].to_numpy()[pred] > 0)
    print(f"== HOLDOUT 2025-01-01 .. 2026-10-31 ({len(t)} half-hours, {len(np.unique(d))} days; up rate {100*t.mean():.1f}%). "
          f"Models trained on days <= 2024-12-31 only.")
    print(f"   yesterday baseline (slot above yesterday's mean): BA {100*bas(t, yday):.3f}, accuracy {100*np.mean(yday == t):.3f}")
    ids, inv = np.unique(d, return_inverse=True)
    s = lambda x: np.bincount(inv, weights=x.astype(float), minlength=len(ids))
    rng = np.random.default_rng(9417); draws = [rng.integers(0, len(ids), len(ids)) for _ in range(4000)]
    allrows = []
    for thr in (0.5, 0.57):
        rows = []
        for k, p in M.items():
            r = {"model": k}
            for lab, m in (("2025-26", np.ones(len(t), bool)), ("2025", yr == 2025), ("2026 Jan-Oct", yr == 2026)):
                r[f"BA {lab}"] = 100 * bas(t[m], p[m] > thr); r[f"acc {lab}"] = 100 * np.mean((p[m] > thr) == t[m])
            r["pred up%"] = 100 * np.mean(p > thr)
            rows.append(r); allrows.append(dict(r, threshold=thr))
        T = pd.DataFrame(rows).set_index("model")
        with pd.option_context("display.width", 240, "display.max_colwidth", 75):
            print(f"\n== Threshold {thr}" + (" (pre-declared)" if thr == 0.5 else " (descriptive only - not a choice)"))
            print(T.round(3).to_string())
        if "CHAMPION (final model)" in M:
            print(f"   Paired day-block bootstrap at {thr}: CHAMPION minus each alternative, BA [95% CI] | accuracy [95% CI]")
            A = M["CHAMPION (final model)"] > thr
            for k in [k for k in M if not k.startswith("CHAMPION")]:
                B = M[k] > thr
                c = np.stack([s(t), s(~t), s(A & t), s(~A & ~t), s(B & t), s(~B & ~t), s(A == t), s(B == t), s(np.ones(len(t)))], 1)
                f = lambda v: (50 * (v[:, 2] / v[:, 0] + v[:, 3] / v[:, 1]) - 50 * (v[:, 4] / v[:, 0] + v[:, 5] / v[:, 1]), 100 * (v[:, 6] - v[:, 7]) / v[:, 8])
                pt = f(c.sum(0)[None]); g, a = f(np.stack([c[dr].sum(0) for dr in draws]))
                print(f"     vs {k[:72]:72s} BA {pt[0][0]:+.3f} [{np.percentile(g, 2.5):+.2f}, {np.percentile(g, 97.5):+.2f}] | "
                      f"acc {pt[1][0]:+.3f} [{np.percentile(a, 2.5):+.2f}, {np.percentile(a, 97.5):+.2f}]")
    A = pd.DataFrame(allrows)
    A.to_csv(f"{H}/holdout_results.csv", index=False)
    W = A.pivot(index="model", columns="threshold", values="BA 2025-26")
    W["0.57 minus 0.5"] = W[0.57] - W[0.5]
    print("\n== Threshold effect on the holdout (BA 2025-26)"); print(W.round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "mlp":
        mlp_worker(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "prep":
        prep()
    elif sys.argv[1] == "prep_neg":
        prep_neg()
    elif sys.argv[1] == "fit":
        with ProcessPoolExecutor(int(os.environ.get("WORKERS", 4))) as pool:
            for n in pool.map(fit_one, jobs()):
                print(time.strftime("%H:%M:%S"), "done", n, flush=True)
    elif sys.argv[1] == "score":
        score()
