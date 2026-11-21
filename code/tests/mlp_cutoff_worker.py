"""Isolated Torch-only helper for test_training_cutoff_leakage.py.

Never loads LightGBM artifacts into the Torch process. All files are temporary.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'forecast_package'))
from model import load_data,market_days,training_table,predict_numpy_mlp
from mlp_train import fit_demand_mlp,predict_demand_mlp


def main(directory):
    columns=json.loads((CODE/'forecast_package/demand_feature_columns.json').read_text())
    fitted=[];numpy_predictions=[];torch_predictions=[];maximum_difference=0.
    for name in ['original','changed']:
        frame=load_data(directory/f'{name}.csv');x,y=training_table(frame);x=x.loc[:,columns]
        days=market_days(x.index);train=days<pd.Timestamp('2016-01-01');test=days==pd.Timestamp('2016-01-01')
        assert days[train].max()==pd.Timestamp('2015-12-31')
        if fitted:
            pd.testing.assert_frame_equal(x.loc[train],fitted[0][0])
            pd.testing.assert_series_equal(y.loc[train,'demand'],fitted[0][1])
            pd.testing.assert_frame_equal(x.loc[test],fitted[0][2])
        bundle=fit_demand_mlp(x.loc[train],y.loc[train,'demand'],epochs=2,seed=9417)
        prediction=predict_demand_mlp(bundle,x.loc[test]);torch_predictions.append(prediction)
        state=bundle['model'].state_dict()
        arrays={key.replace('.','_'):value.detach().cpu().numpy() for key,value in state.items()}
        arrays.update(mean=bundle['scaler'].mean_,scale=bundle['scaler'].scale_,features=np.array(columns))
        path=directory/f'{name}_export.npz';np.savez_compressed(path,**arrays)
        with np.load(path,allow_pickle=False) as saved:
            exported={key:saved[key] for key in saved.files}
        numpy_prediction=predict_numpy_mlp(exported,x.loc[test]);numpy_predictions.append(numpy_prediction)
        np.testing.assert_allclose(prediction,numpy_prediction,rtol=1e-6,atol=.002)
        maximum_difference=max(maximum_difference,float(np.max(np.abs(prediction-numpy_prediction))))
        fitted.append((x.loc[train],y.loc[train,'demand'],x.loc[test],bundle,exported))
    a,b=fitted[0][3],fitted[1][3]
    for key in ['mean_','var_','scale_','n_samples_seen_']:
        np.testing.assert_array_equal(getattr(a['scaler'],key),getattr(b['scaler'],key))
    for key,value in a['model'].state_dict().items():
        assert torch.equal(value,b['model'].state_dict()[key]),key
    for key,value in fitted[0][4].items():np.testing.assert_array_equal(value,fitted[1][4][key])
    np.testing.assert_array_equal(torch_predictions[0],torch_predictions[1])
    np.testing.assert_array_equal(numpy_predictions[0],numpy_predictions[1])
    assert 'lightgbm' not in sys.modules
    (directory/'report.json').write_text(json.dumps(dict(training_features_equal=True,training_targets_equal=True,
        scalers_equal=True,weights_equal=True,torch_predictions_equal=True,numpy_exports_equal=True,
        numpy_predictions_equal=True,numpy_torch_max_difference_MW=maximum_difference),indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main(Path(sys.argv[1]))
