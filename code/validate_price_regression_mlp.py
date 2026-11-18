"""Actual-price MLPs and fixed ensembles, selected without 2014 tuning."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parent / "forecast_package"))
from model import load_data,market_days,training_table
from price_regression_mlp import fit_price_regression_mlp,predict_price_regression_mlp

ROOT = Path(__file__).resolve().parents[1]


def calibrator_inputs(margin):
    margin = np.asarray(margin,dtype=float)
    return (np.sign(margin)*np.log1p(np.abs(margin))).reshape(-1,1)


def fit_calibrator(margin,labels):
    model = make_pipeline(StandardScaler(),LogisticRegression(
        C=1.,class_weight="balanced",max_iter=1000,random_state=9417))
    model.fit(calibrator_inputs(margin),labels)
    return model


def main():
    started = time.monotonic()
    output = ROOT / "results" / "price_regression_mlp_2014"
    output.mkdir(parents=True,exist_ok=False)
    source = ROOT.parent / "COMP9417_price_regression" / "results"
    source_metadata = json.loads((source / "price_regression_2014/metadata.json").read_text())
    assert not source_metadata["quick"] and source_metadata["training_end"] == "2013-12-31"
    columns = json.loads((ROOT / "code/price_feature_columns.json").read_text())
    frame = load_data(ROOT / "data/nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2015-01-01 00:00:00"]
    features,labels = training_table(frame)
    features = features.loc[:,columns]
    days = market_days(features.index)
    means = frame.nsw_price.groupby(market_days(frame.index)).mean().shift(1)
    predictions, audit = [], []
    for year in (2011,2012,2013,2014):
        train,test = days.year<year,days.year==year
        assert days[train].max()<days[test].min()
        cached_path = (source / f"price_stacking_2014/oof_{year}.csv" if year<2014
                       else source / "price_regression_2014/predictions.csv")
        cached = pd.read_csv(cached_path,index_col="timestamp",parse_dates=True)
        assert cached.index.equals(features.index[test])
        np.testing.assert_array_equal(cached.actual_up,labels.loc[test,"price"])
        print(f"Training price MLP for {year}, data through {days[train].max().date()}",flush=True)
        bundle = fit_price_regression_mlp(features.loc[train],
            frame.loc[features.index[train],"nsw_price"],checkpoints=(5,10))
        threshold = pd.Series(days[test],index=features.index[test]).map(means)
        fold = pd.DataFrame({"actual_price":frame.loc[features.index[test],"nsw_price"],
            "actual_up":labels.loc[test,"price"],"previous_day_mean":threshold,
            "LGBM_regressed_price":cached.nsw_price_pred,
            "classifier_probability":cached.classifier_probability},index=features.index[test])
        for epoch in (5,10):
            fold[f"MLP_{epoch}_price"] = predict_price_regression_mlp(bundle,features.loc[test],epoch)
            if year==2014:
                torch.save(bundle["states"][epoch],output / f"weights_{epoch}.pt")
        if year==2014:
            joblib.dump({"scaler":bundle["scaler"],"features":columns},output / "preprocessing.joblib")
        fold.to_csv(output / f"forecasts_{year}.csv",index_label="timestamp")
        predictions.append(fold)
        audit.append({"forecast_year":year,"training_end":str(days[train].max().date()),
                      "training_rows":int(train.sum()),"history":bundle["history"]})
    oof = pd.concat(predictions[:3]).sort_index()
    valid = predictions[3].copy()
    assert oof.index.max()<valid.index.min()
    labels_valid = valid.actual_up.to_numpy()
    rows = []
    def report(name,predicted,threshold=None,actual_price=None):
        row = {"model":name,"threshold":threshold,
               "accuracy":float(np.mean(labels_valid==np.asarray(predicted))*100),
               "balanced_accuracy":float(balanced_accuracy_score(labels_valid,predicted)*100)}
        if actual_price is not None:
            errors=np.asarray(actual_price)-valid.actual_price.to_numpy()
            row.update(price_MAE=float(np.mean(np.abs(errors))),price_RMSE=float(np.sqrt(np.mean(errors**2))))
        rows.append(row)
    for threshold in (.5,.57):
        report(f"current_classifier_{threshold}",valid.classifier_probability>threshold,threshold)
    report("LightGBM_regression",valid.LGBM_regressed_price>valid.previous_day_mean,
           actual_price=valid.LGBM_regressed_price)
    calibrated = {}
    for name,column in (("LightGBM","LGBM_regressed_price"),("MLP_5","MLP_5_price"),("MLP_10","MLP_10_price")):
        calibration = fit_calibrator(oof[column]-oof.previous_day_mean,oof.actual_up)
        calibrated[name] = calibration.predict_proba(calibrator_inputs(valid[column]-valid.previous_day_mean))[:,1]
        joblib.dump(calibration,output / f"calibrator_{name}.joblib")
    for epoch in (5,10):
        price = valid[f"MLP_{epoch}_price"]
        report(f"MLP_{epoch}_regression",price>valid.previous_day_mean,actual_price=price)
        blended_price = .5*valid.LGBM_regressed_price+.5*price
        report(f"50_50_LGBM_MLP_{epoch}_price",blended_price>valid.previous_day_mean,actual_price=blended_price)
        # Fixed averaging of classifier probability and calibrated regression signal.
        blend_probability = .5*valid.classifier_probability.to_numpy()+.5*calibrated[f"MLP_{epoch}"]
        valid[f"classifier_MLP_{epoch}_blend_probability"] = blend_probability
        for threshold in (.5,.57):
            report(f"50_50_classifier_MLP_{epoch}_probability_{threshold}",blend_probability>threshold,threshold)
    thirds = (valid.classifier_probability.to_numpy()+calibrated["MLP_10"]+calibrated["LightGBM"])/3
    for threshold in (.5,.57):
        report(f"equal_thirds_classifier_LGBMreg_MLP10_{threshold}",thirds>threshold,threshold)
    comparison=pd.DataFrame(rows)
    comparison.to_csv(output / "metrics.csv",index=False)
    valid.to_csv(output / "predictions_2014.csv",index_label="timestamp")
    metadata={"architecture":[571,128,64,1],"epochs":[5,10],"target":"Actual NSW price divided by 100",
        "loss":"L1, equal observation weights; no price clipping","seed":9417,
        "calibration":"Balanced logistic regression on signed-log predicted margins, fitted on chronological 2011–2013 predictions",
        "probability_blend_weights":"Fixed 50/50 or equal thirds; thresholds fixed at 0.5 and 0.57",
        "folds":audit,"evaluation_year":2014,"holdout_evaluated":False,
        "torch":torch.__version__,"seconds_total":time.monotonic()-started}
    (output / "metadata.json").write_text(json.dumps(metadata,indent=2))
    assert "lightgbm" not in sys.modules
    print(comparison.to_string(index=False),flush=True)
    print(f"Completed in {metadata['seconds_total']:.1f}s",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
