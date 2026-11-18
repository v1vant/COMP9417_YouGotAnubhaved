"""Recreate one selected RF, then assess 2014 complementarity; five-minute cap."""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def worker(output):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from threadpoolctl import threadpool_limits

    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("rf_reference_features",output / "reference_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = module.load_data(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2015-01-01 00:00:00"]
    features,labels = module.training_table(frame)
    assert features.shape[1] == 240
    days = module.market_days(features.index)
    train,valid = days.year<=2013,days.year==2014
    config = json.loads((output / "config.json").read_text())
    params = config["selected"]["price"]["config"]
    model = RandomForestClassifier(n_estimators=params["trees"],max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],max_features=params["max_features"],
        bootstrap=True,class_weight="balanced",n_jobs=2,random_state=9417)
    print(f"Fitting selected RF: {int(train.sum())} rows, 240 features, 200 trees",flush=True)
    with threadpool_limits(limits=2):
        model.fit(features.loc[train],labels.loc[train,"price"])
        probability = model.predict_proba(features.loc[valid])[:,model.classes_.tolist().index(1)]
    print(f"Fit and predictions complete after {time.monotonic()-started:.1f}s",flush=True)
    source = ROOT.parent / "COMP9417_price_regression/results/price_regression_2014/predictions.csv"
    current = pd.read_csv(source,index_col="timestamp",parse_dates=True)
    assert current.index.equals(features.index[valid])
    actual = labels.loc[valid,"price"].to_numpy()
    np.testing.assert_array_equal(current.actual_up,actual)
    lp = current.classifier_probability.to_numpy()
    rows = []
    for name,p,threshold in (("current_LightGBM",lp,.57),("LightGBM_threshold_0.5",lp,.5),
                             ("random_forest",probability,params["threshold"]),
                             ("blend_50_50_threshold_0.5",.5*(lp+probability),.5),
                             ("blend_50_50_threshold_0.57",.5*(lp+probability),.57)):
        predicted = p>threshold
        rows.append({"model":name,"threshold":threshold,
            "accuracy":float(np.mean(actual==predicted)*100),
            "balanced_accuracy":float(balanced_accuracy_score(actual,predicted)*100)})
    lc = (lp>.57)==actual
    rc = (probability>params["threshold"])==actual
    disagreement = (lp>.57)!=(probability>params["threshold"])
    summary = {"rows":len(actual),"training_rows":int(train.sum()),"features":240,
        "both_correct":int((lc&rc).sum()),"LightGBM_only_correct":int((lc&~rc).sum()),
        "RF_only_correct":int((~lc&rc).sum()),"both_wrong":int((~lc&~rc).sum()),
        "RF_corrects_LightGBM_errors_percent":float(rc[~lc].mean()*100),
        "RF_wins_disagreements_percent":float(rc[disagreement].mean()*100),
        "disagreement_percent":float(disagreement.mean()*100),
        "error_correlation":float(np.corrcoef(~lc,~rc)[0,1]),
        "seconds":time.monotonic()-started,"holdout_evaluated":False,
        "note":"Original RF feature pipeline; fixed selected parameters and threshold. LightGBM uses current 571 features. No blend tuning."}
    pd.DataFrame(rows).to_csv(output / "metrics.csv",index=False)
    pd.DataFrame({"actual_up":actual,"RF_probability":probability,"LightGBM_probability":lp},
        index=features.index[valid]).to_csv(output / "predictions.csv",index_label="timestamp")
    joblib.dump({"estimator":model,"features":list(features.columns),"config":params,
                 "training_end":"2013-12-31"},output / "rf_2013.joblib")
    (output / "complementarity.json").write_text(json.dumps(summary,indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
        return
    output = ROOT / "results/price_rf_complementarity_2014"
    output.mkdir(parents=True,exist_ok=False)
    for source,target in (("code/forecast_package/model.py","reference_model.py"),
                          ("results/random_forest/selection.json","config.json")):
        (output / target).write_bytes(subprocess.check_output(["git","show",f"origin/random_forest:{source}"],cwd=ROOT))
    try:
        subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),"--worker",str(output)],
                       check=True,timeout=300)
    except subprocess.TimeoutExpired:
        (output / "timeout.json").write_text(json.dumps({"status":"stopped","limit_seconds":300}))
        print("Stopped RF experiment at five-minute cap.",flush=True)


if __name__=="__main__":
    main()
