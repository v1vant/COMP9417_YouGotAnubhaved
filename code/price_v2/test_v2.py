import unittest
import numpy as np
import pandas as pd
from model import feature_frame, forecast_features, MARKET, WEATHER
from pipeline import table
from price_context import forecast_price_features


def synthetic(days=75):
    idx=pd.date_range('2013-12-01 00:30',periods=days*48,freq='30min')
    a=np.arange(len(idx));f=pd.DataFrame(index=idx)
    for i,c in enumerate(MARKET):f[c]=5000+a*.01+np.sin(a/20)*100 if c.endswith('demand') else 40+np.sin(a/10)*20
    for i,c in enumerate(WEATHER):f[c]=20+np.cos(a/40)
    return f


class V2Tests(unittest.TestCase):
    def test_schema_parity_and_causality(self):
        f=synthetic();x,y=table(f,context=True)
        self.assertEqual(x.shape[1],590)
        self.assertEqual(len(x)%48,0)
        idx=f.index[-48:];weather=f.loc[idx,WEATHER];history=f.iloc[:-48]
        base=forecast_features(history,weather)
        live=forecast_price_features(history,weather,base,'both')
        pd.testing.assert_frame_equal(x.loc[idx],live)
        changed=f.copy();changed.loc[idx,MARKET]=-1e9
        altered,_=table(changed,context=True)
        pd.testing.assert_frame_equal(x.loc[idx],altered.loc[idx])
        compact,_=table(f,compact=True,context=True)
        self.assertEqual(compact.shape[1],259)

    def test_strict_target(self):
        f=synthetic();f.nsw_price=85.04
        _,y=table(f)
        self.assertTrue((y.price==0).all())

if __name__=='__main__':unittest.main()

class DemandPreservationTests(unittest.TestCase):
    def test_reference_and_v2_demand_are_identical(self):
        import json
        from pathlib import Path
        from model import Forecaster
        from reference_model import Forecaster as Reference
        class Demand:
            def predict(self,x):return np.full(len(x),100.)
        class Price:
            def predict_proba(self,x):return np.tile([.4,.6],(len(x),1))
        columns=json.loads((Path(__file__).parent/'demand_feature_columns.json').read_text())
        f=synthetic(); history=f.iloc[:-48];weather=f.iloc[-48:][WEATHER]
        weights=dict(features=np.array(columns),mean=np.zeros(405),scale=np.ones(405),
            network_0_weight=np.zeros((128,405)),network_0_bias=np.zeros(128),
            network_2_weight=np.zeros((64,128)),network_2_bias=np.zeros(64),
            network_4_weight=np.zeros((1,64)),network_4_bias=np.zeros(1))
        models=dict(demand=dict(task='demand',estimator=Demand(),features=columns),
                    price=dict(task='price',estimator=Price(),features=list(feature_frame(f)),threshold=.57,config={}))
        a=Reference.__new__(Reference);a.models=models;a.demand_mlp=weights
        b=Forecaster.__new__(Forecaster);b.models=models;b.demand_mlp=weights
        pd.testing.assert_frame_equal(a.predict(history,weather),b.predict(history,weather))

    def test_stacking_features_work_without_future_demand(self):
        from stack import demand_features
        f=synthetic(); q=f.nsw_demand.to_numpy().copy();f.loc[f.index[-48:],'nsw_demand']=np.nan
        extra=demand_features(f,q,np.zeros(len(q)))
        self.assertTrue(np.isfinite(extra.iloc[-48:].to_numpy()).all())

class HoldoutProtocolTests(unittest.TestCase):
    def test_holdout_keeps_2025_model_frozen_and_updates_in_2026(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from holdout import evaluate_policy
        index=pd.date_range('2013-01-01 00:30','2026-11-01 00:00',freq='30min')
        x=pd.DataFrame({'feature':np.ones(len(index))},index=index)
        y=pd.DataFrame({'price':np.arange(len(index))%2},index=index)
        cuts=[]
        def fitted(config,features,labels,cutoff):
            cuts.append(cutoff)
            return {'training_start':'2013-01-01','training_end':str(cutoff-pd.Timedelta(days=1))}
        with tempfile.TemporaryDirectory() as folder, patch('holdout.table',return_value=(x,y)),patch('holdout.fit',side_effect=fitted),patch('holdout.predict_price_probability',side_effect=lambda m,z:np.full(len(z),.6)):
            result=evaluate_policy(None,{'config':{'threshold':.57},'refresh':False},Path(folder))
        self.assertEqual(cuts,[pd.Timestamp('2024-01-01'),pd.Timestamp('2026-01-01')])
        self.assertEqual(len(result),669*48)
        self.assertEqual(result.index.min(),pd.Timestamp('2025-01-01 00:30'))
        self.assertEqual(result.index.max(),pd.Timestamp('2026-11-01 00:00'))
        self.assertTrue((result.prediction==1).all())
