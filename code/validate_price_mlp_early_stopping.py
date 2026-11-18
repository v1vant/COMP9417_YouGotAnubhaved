"""Select price MLP epochs on 2013; refit through 2013 and evaluate 2014."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent / "forecast_package"))
from model import load_data, market_days, training_table
from price_mlp import PriceMLP, fit_price_mlp, predict_price_probability

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    output = ROOT / "results" / "price_mlp_early_stopping_2014"
    output.mkdir(parents=True, exist_ok=False)
    columns = json.loads((ROOT / "code" / "price_feature_columns.json").read_text())
    frame = load_data(ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    frame = frame.loc[:"2015-01-01 00:00:00"]
    features, labels = training_table(frame)
    features = features.loc[:, columns]
    days = market_days(features.index)
    train, select = days.year <= 2012, days.year == 2013
    selection = fit_price_mlp(features.loc[train], labels.loc[train,"price"],
        epochs=100, validation=(features.loc[select],labels.loc[select,"price"]),patience=10)
    epochs = selection["best_epoch"]
    assert 1<=epochs<=100
    pd.DataFrame(selection["history"]).to_csv(output / "selection_history_2013.csv",index=False)
    print(f"Selected {epochs} epochs; refitting through 2013",flush=True)
    refit, valid = days.year <= 2013, days.year == 2014
    bundle = fit_price_mlp(features.loc[refit],labels.loc[refit,"price"],epochs=epochs)
    probability = predict_price_probability(bundle,features.loc[valid])
    assert np.isfinite(probability).all()
    actual = labels.loc[valid,"price"].to_numpy()
    rows = []
    for threshold in (0.5,0.57):
        predicted = probability>threshold
        rows.append({"model":f"selected_{epochs}_epoch_MLP_threshold_{threshold}",
                     "accuracy":float(np.mean(predicted==actual)*100),
                     "balanced_accuracy":float(balanced_accuracy_score(actual,predicted)*100)})
    previous = pd.read_csv(ROOT / "results" / "price_mlp_5epoch_2014" / "metrics.csv")
    comparison = pd.concat([previous,pd.DataFrame(rows)],ignore_index=True)
    comparison.to_csv(output / "metrics.csv",index=False)
    pd.DataFrame({"actual_up":actual,"probability":probability},index=features.index[valid]).to_csv(
        output / "predictions.csv",index_label="timestamp")
    torch.save(bundle["model"].state_dict(),output / "weights.pt")
    joblib.dump({"scaler":bundle["scaler"],"features":columns},output / "preprocessing.joblib")
    reloaded=PriceMLP(len(columns))
    reloaded.load_state_dict(torch.load(output / "weights.pt",weights_only=True,map_location="cpu"))
    np.testing.assert_allclose(probability,predict_price_probability(dict(bundle,model=reloaded),features.loc[valid]))
    metadata={"selection_training_years":[2010,2011,2012],"selection_year":2013,
        "selection_metric":"balanced accuracy at fixed 0.5 threshold","max_epochs":100,"patience":10,
        "selected_epochs":epochs,"selection_epochs_run":len(selection["history"]),
        "best_selection_balanced_accuracy":selection["best_validation_balanced_accuracy"],
        "refit_training_years":[2010,2011,2012,2013],"evaluation_year":2014,
        "class_weights":bundle["class_weights"],"state_dict_reload_passed":True,
        "holdout_evaluated":False,"seconds_total":time.monotonic()-started}
    (output / "metadata.json").write_text(json.dumps(metadata,indent=2))
    assert "lightgbm" not in sys.modules
    print(comparison.to_string(index=False),flush=True)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=2):
        main()
