"""Repeat the team's day-type silhouette (origin/main code/analysis_tools.py, day_type_analysis) on 2014 for k = 2, 3, 4.
Their steps: shape = day / day mean; PCA(0.95, full) fit on 2010-2013; KMeans(k, n_init=20, random_state=9417) on the
2010-2013 PCA scores; silhouette_score(scores, labels, sample_size=min(1000, n), random_state=9417).
2014: transform with the same PCA, assign with the same KMeans (predict), silhouette on the 2014 scores.
Only data up to market day 2014-12-31 is read: the first 1826 x 48 rows of the CSV, so 2015 onwards and the 2025-26
holdout are never loaded. Run from the repo root: python code/day_type_silhouette_2014.py"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analysis_tools as team                     # the team's day-type code, unmodified
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

raw = pd.read_csv(os.path.join(ROOT, "data", "nem_weather_halfhourly_2010_2026.csv"), usecols=["timestamp", "nsw_demand"],
                  parse_dates=["timestamp"], nrows=1826 * 48).set_index("timestamp")
assert raw.index[0] == pd.Timestamp("2010-01-01 00:30") and raw.index[-1] == pd.Timestamp("2015-01-01 00:00")

# 1) their function as-is: must reproduce their saved selection (results/day_type_selection.json on main)
_, _, _, info = team.day_type_analysis(raw)
saved = {2: 0.4124379914266957, 3: 0.3432045307828315, 4: 0.3109008954868571, 5: 0.33516125708861694, 6: 0.3235533169722595}
assert all(abs(info["silhouette_scores"][k] - saved[k]) < 1e-12 for k in saved), info["silhouette_scores"]
print("team function reproduces the saved 2010-2013 silhouettes exactly; selected k =", info["selected_clusters"])

# 2) same steps, keeping the k = 2, 3, 4 models, then score 2014
days = (raw.index - pd.Timedelta(minutes=30)).normalize().unique()
profiles = raw["nsw_demand"].to_numpy().reshape(-1, 48)
shapes = profiles / profiles.mean(axis=1, keepdims=True)
tr, te = days.year <= 2013, days.year == 2014
pca = PCA(n_components=0.95, svd_solver="full").fit(shapes[tr])
assert np.allclose(pca.components_, info["pca"].components_)
Ztr, Zte = pca.transform(shapes[tr]), pca.transform(shapes[te])
rows = []
for k in (2, 3, 4):
    km = KMeans(n_clusters=k, n_init=20, random_state=team.SEED).fit(Ztr)
    lab_tr, lab_te = km.labels_, km.predict(Zte)
    s_tr_team = silhouette_score(Ztr, lab_tr, sample_size=min(1000, len(Ztr)), random_state=team.SEED)
    assert abs(s_tr_team - saved[k]) < 1e-12
    rows.append({"k": k,
                 "silhouette 2010-13 (team, 1000-day sample)": s_tr_team,
                 "silhouette 2010-13 (all 1461 days)": silhouette_score(Ztr, lab_tr),
                 "silhouette 2014 (365 days, same PCA + KMeans)": silhouette_score(Zte, lab_te),
                 "2014 days per cluster": "/".join(map(str, np.bincount(lab_te, minlength=k)))})
T = pd.DataFrame(rows).set_index("k")
pd.set_option("display.width", 200)
print(T.round(4).to_string())
print("best k on 2014:", T["silhouette 2014 (365 days, same PCA + KMeans)"].idxmax())
out = os.path.join(ROOT, "results", "day_type_silhouette_2014")
os.makedirs(out, exist_ok=True)
T.to_csv(os.path.join(out, "silhouette_2014.csv"))
