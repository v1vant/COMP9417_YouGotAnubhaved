"""Hierarchical (Ward) clustering of NSW daily demand SHAPES, k = 2..12, fitted on 2010-2013, validated on 2014.
Pipeline as the team's day_type_analysis: each day's 48 values / the day's mean; PCA (95% variance) fitted on 2010-2013.
2014 days are assigned to the nearest 2010-2013 cluster centroid in PCA space.
Pre-declared rule: choose the LARGEST k whose prediction strength on 2014 >= 0.8 (Tibshirani & Walther 2005).
Only data up to 2014-12-31 is read. Run from the repo root: python code/day_types/validate_k_hierarchical.py"""
import json, os, sys, warnings
import numpy as np, pandas as pd, holidays
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
warnings.filterwarnings("ignore")
from common import load_demand, OUT
os.makedirs(OUT, exist_ok=True)
KS = range(2, 13); THRESH = 0.8; SEED = 9417

df = load_demand("2014-12-31").reset_index()
df["day"] = (df.timestamp - pd.Timedelta(minutes=30)).dt.normalize()
df["slot"] = ((df.timestamp - pd.Timedelta(minutes=30)) - df.day).dt.total_seconds() // 1800
D = df.pivot(index="day", columns="slot", values="nsw_demand")
assert D.shape[1] == 48 and not D.isna().any().any() and D.index.max() == pd.Timestamp("2014-12-31")
L = D.mean(axis=1); S = D.div(L, axis=0).to_numpy()
tr = np.asarray(D.index.year <= 2013); va = np.asarray(D.index.year == 2014)
pca = PCA(n_components=0.95, svd_solver="full").fit(S[tr])
Ztr, Zva = pca.transform(S[tr]), pca.transform(S[va])
print(f"days: train {tr.sum()} (2010-2013), validation {va.sum()} (2014); PCA components {pca.n_components_}, "
      f"variance {pca.explained_variance_ratio_.sum():.3f}")

Htr = linkage(Ztr, "ward"); Hva = linkage(Zva, "ward")


def cut(H, k):
    lab = fcluster(H, k, "maxclust") - 1
    assert len(np.unique(lab)) == k
    return lab


def assign(Z, cent):
    d2 = ((Z[:, None, :] - cent[None]) ** 2).sum(-1)
    return d2.argmin(1), d2.min(1)


def prediction_strength(test_lab, pred_lab):
    """min over test clusters (size >= 2) of the share of within-cluster pairs that pred_lab also puts together."""
    out = []
    for c in np.unique(test_lab):
        idx = np.where(test_lab == c)[0]
        n = len(idx)
        if n < 2:
            continue
        counts = np.bincount(pred_lab[idx])
        out.append(((counts * (counts - 1)).sum() / (n * (n - 1)), n))
    return min(o[0] for o in out), out


# week blocks of 2014 for a bootstrap of prediction strength
weeks = np.asarray((D.index[va] - pd.Timestamp("2014-01-01")).days // 7)
rng = np.random.default_rng(SEED)
rows, per_k = [], {}
q1 = None
for k in KS:
    lab = cut(Htr, k)
    cent = np.stack([Ztr[lab == c].mean(0) for c in range(k)])
    pred, d2va = assign(Zva, cent)
    d2tr = ((Ztr - cent[lab]) ** 2).sum(1)
    test = cut(Hva, k)
    ps, detail = prediction_strength(test, pred)
    # bootstrap: resample 2014 weeks, recluster the resample, recompute PS
    bs = []
    for _ in range(200):
        wk = rng.choice(np.unique(weeks), len(np.unique(weeks)), replace=True)
        idx = np.concatenate([np.where(weeks == w)[0] for w in wk])
        Zb = Zva[idx] + rng.normal(0, 1e-9, Zva[idx].shape)      # jitter only to break exact duplicate ties
        tb = fcluster(linkage(Zb, "ward"), k, "maxclust") - 1
        if len(np.unique(tb)) < k:
            continue
        bs.append(prediction_strength(tb, pred[idx])[0])
    occ_tr = np.bincount(lab, minlength=k); occ_va = np.bincount(pred, minlength=k)
    sil_va = silhouette_score(Zva, pred) if len(np.unique(pred)) > 1 else np.nan
    sil_tr = silhouette_score(Ztr, lab)
    rows.append({"k": k, "PS_2014": ps, "PS_boot_p10": np.percentile(bs, 10), "PS_boot_median": np.median(bs),
                 "share_boot_PS>=0.8": np.mean(np.array(bs) >= THRESH),
                 "PS_mean_over_clusters": np.mean([o[0] for o in detail]),
                 "ARI(own 2014 clustering, assigned)": adjusted_rand_score(test, pred),
                 "silhouette_train": sil_tr, "silhouette_2014_assigned": sil_va,
                 "WCSS_ratio_2014/train": d2va.mean() / d2tr.mean(),
                 "min_cluster_days_train": occ_tr.min(), "min_cluster_days_2014": occ_va.min(),
                 "empty_clusters_2014": int((occ_va == 0).sum())})
    per_k[k] = {"labels_train": lab, "pred_2014": pred, "cent": cent, "detail": detail}
T = pd.DataFrame(rows).set_index("k")
# within-cluster sum of squares drop per added cluster, train vs 2014
T["mean_sq_dist_train"] = [((Ztr - per_k[k]["cent"][per_k[k]["labels_train"]]) ** 2).sum(1).mean() for k in KS]
T["mean_sq_dist_2014"] = [assign(Zva, per_k[k]["cent"])[1].mean() for k in KS]
pd.set_option("display.width", 250)
print("\nValidation on 2014 (2010-2013 Ward clusters, 2014 days assigned to nearest centroid)")
print(T.round(3).to_string())
ok = [k for k in KS if T.loc[k, "PS_2014"] >= THRESH]
chosen = max(ok) if ok else None
print(f"\nPRE-DECLARED RULE (largest k with PS_2014 >= {THRESH}): k = {chosen}  (k passing: {ok})")
T.to_csv(os.path.join(OUT, "hierarchical_validation_by_k.csv"))

# interpretation helper for any k
au = holidays.Australia(subdiv="NSW", years=range(2010, 2015))
nonwork = np.array([(x in au) or x.weekday() >= 5 for x in D.index])
month = D.index.month.to_numpy()
blocks = {"00-06": (0, 12), "06-10": (12, 20), "10-16": (20, 32), "16-21": (32, 42), "21-24": (42, 48)}


def describe(k):
    lab_all = np.empty(len(D), int); lab_all[tr] = per_k[k]["labels_train"]; lab_all[va] = per_k[k]["pred_2014"]
    out = []
    for c in range(k):
        m = lab_all == c
        prof = S[m & tr].mean(0)
        r = {"cluster": c, "days_2010_13": int((m & tr).sum()), "days_2014": int((m & va).sum()),
             "nonwork_%": 100 * nonwork[m].mean(), "mean_MW": L.to_numpy()[m].mean(),
             "top_months": ",".join(str(x) for x in pd.Series(month[m]).value_counts().index[:3]),
             "summer_%": 100 * np.isin(month[m], [12, 1, 2]).mean(), "winter_%": 100 * np.isin(month[m], [6, 7, 8]).mean(),
             "peak_h": np.argmax(prof) / 2, "peak/mean": prof.max(), "min/mean": prof.min()}
        r.update({f"{b}%": 100 * prof[a:z].sum() / prof.sum() for b, (a, z) in blocks.items()})
        out.append(r)
    return pd.DataFrame(out).set_index("cluster")


for k in sorted(set([2, chosen] + ([chosen + 1] if chosen and chosen < 12 else []))):
    if k is None:
        continue
    print(f"\nClusters at k = {k} (profile stats from 2010-2013 members; day counts 2010-13 and 2014)")
    print(describe(k).round(2).to_string())
    describe(k).to_csv(os.path.join(OUT, f"hierarchical_clusters_k{k}.csv"))
json.dump({"pca_components": int(pca.n_components_), "chosen_k": chosen, "k_passing": ok, "threshold": THRESH},
          open(os.path.join(OUT, "hierarchical_summary.json"), "w"), indent=1)
