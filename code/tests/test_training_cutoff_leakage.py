"""Fast retraining invariance checks on the real walk-forward cutoff path.

Future market values are perturbed from the forecast day onward. Weather on
that day stays fixed because it is permitted input; weather after it changes.
These tests do not fit or overwrite the full-data production artifacts.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'forecast_package'))
import run_experiments as experiments
import train as training
from c2_price import price_table
from model import MARKET,WEATHER,market_days,training_table
from test_forecasting import synthetic_data


class TrainingCutoffLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=synthetic_data(days=90,start='2015-11-01')
        cls.forecast_day=pd.Timestamp('2016-01-01')
        cls.cutoff=cls.forecast_day-pd.Timedelta(days=1)
        cls.changed=cls.original.copy()
        days=market_days(cls.original.index)
        future_market=days>=cls.forecast_day
        later_weather=days>cls.forecast_day
        for number,column in enumerate(MARKET):
            cls.changed.loc[future_market,column]=500_000+number*1000+np.arange(future_market.sum())
        for number,column in enumerate(WEATHER):
            cls.changed.loc[later_weather,column]=-50_000-number*100-np.arange(later_weather.sum())
        pd.testing.assert_frame_equal(cls.original.loc[days<=cls.cutoff],cls.changed.loc[days<=cls.cutoff])
        pd.testing.assert_frame_equal(cls.original.loc[days==cls.forecast_day,WEATHER],
                                      cls.changed.loc[days==cls.forecast_day,WEATHER])
        assert not cls.original.loc[future_market,MARKET].equals(cls.changed.loc[future_market,MARKET])
        assert not cls.original.loc[later_weather,WEATHER].equals(cls.changed.loc[later_weather,WEATHER])
        cls.before=cls.artifact_hashes()

    @staticmethod
    def artifact_hashes():
        return {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                for directory in ['forecast_package','submission_package']
                for path in (CODE/directory/'artifacts').iterdir() if path.is_file()}

    @classmethod
    def tearDownClass(cls):
        if cls.before!=cls.artifact_hashes():
            raise AssertionError('Leakage tests must not modify production artifacts')

    def check_retraining(self,task,table_builder,config):
        fits=[];outputs=[];tables=[]
        for frame in [self.original,self.changed]:
            x,y=table_builder(frame)
            # Compare to a physically truncated raw-data build as well.
            prefix=frame.loc[market_days(frame.index)<=self.cutoff]
            prefix_x,prefix_y=table_builder(prefix)
            past=market_days(x.index)<=self.cutoff
            pd.testing.assert_frame_equal(x.loc[past],prefix_x)
            pd.testing.assert_frame_equal(y.loc[past],prefix_y)
            keep=market_days(x.index)<=self.forecast_day
            x,y=x.loc[keep],y.loc[keep]
            captured=[]

            def fit_and_record(actual_task,actual_config,features,truth):
                # A spy around the real fitter: no fake model or prediction.
                self.assertEqual(actual_task,task)
                self.assertEqual(market_days(features.index).max(),self.cutoff)
                self.assertTrue((market_days(features.index)<self.forecast_day).all())
                bundle=training.fit_model(actual_task,actual_config,features,truth)
                captured.append((features.copy(),truth.copy(),bundle))
                return bundle

            with threadpool_limits(limits=2),patch.object(experiments,'fit_model',side_effect=fit_and_record):
                prediction,audit,_=experiments.simulate_policy(task,'periodic_expanding',None,config,x,y,
                    pd.DatetimeIndex([self.forecast_day]),pd.Series([0.],index=[self.forecast_day]),
                    np.zeros(365),np.zeros(365))
            self.assertEqual(len(captured),1)
            self.assertEqual(audit[0]['training_end'],'2015-12-31')
            self.assertEqual(audit[0]['effective_day'],'2016-01-01')
            fits.append(captured[0]);outputs.append(prediction);tables.append((x,y))
        pd.testing.assert_frame_equal(fits[0][0],fits[1][0])
        pd.testing.assert_frame_equal(fits[0][1],fits[1][1])
        eval_rows=market_days(tables[0][0].index)==self.forecast_day
        pd.testing.assert_frame_equal(tables[0][0].loc[eval_rows],tables[1][0].loc[eval_rows])
        self.assertFalse(tables[0][1].loc[eval_rows].equals(tables[1][1].loc[eval_rows]))
        np.testing.assert_array_equal(outputs[0],outputs[1])
        first,second=fits[0][2],fits[1][2]
        self.assertEqual(first['features'],second['features'])
        self.assertEqual(first['estimator'].get_params(),second['estimator'].get_params())
        # Exact trees, not just matching class decisions.
        self.assertEqual(first['estimator'].booster_.model_to_string(),
                         second['estimator'].booster_.model_to_string())
        if task=='price':
            for bundle,x in [(first,tables[0][0]),(second,tables[1][0])]:
                self.assertEqual(len(bundle['features']),597)
            a=first['estimator'].predict_proba(tables[0][0].loc[eval_rows,first['features']])
            b=second['estimator'].predict_proba(tables[1][0].loc[eval_rows,second['features']])
            np.testing.assert_array_equal(a,b)

    def test_demand_retraining_ignores_future_perturbations(self):
        self.check_retraining('demand',training_table,
            dict(family='boosted',trees=8,leaves=7,learning_rate=.1))

    def test_c2_retraining_ignores_future_perturbations(self):
        self.check_retraining('price',price_table,
            dict(family='c2',seeds=[9417],threshold=.5,weather_shape=True,
                 params=dict(n_estimators=8,num_leaves=7,learning_rate=.1,
                             colsample_bytree=.5,min_child_samples=160)))

    def test_deliberate_lookahead_is_detected(self):
        tables=[]
        for frame in [self.original,self.changed]:
            x,_=training_table(frame)
            # Deliberate bug: tomorrow's realised demand leaks into training.
            x=x.copy();x['deliberately_leaky_next_day_demand']=frame.nsw_demand.shift(-48).loc[x.index]
            tables.append(x.loc[market_days(x.index)<=self.cutoff])
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(tables[0],tables[1])

    @unittest.skipIf(importlib.util.find_spec('torch') is None,'PyTorch required for isolated MLP training test')
    def test_mlp_scaler_weights_and_export_ignore_future_perturbations(self):
        with tempfile.TemporaryDirectory(prefix='mlp_cutoff_test_') as directory:
            path=Path(directory)
            for name,frame in [('original',self.original),('changed',self.changed)]:
                frame.to_csv(path/f'{name}.csv',index_label='timestamp')
            result=subprocess.run([sys.executable,str(CODE/'tests/mlp_cutoff_worker.py'),str(path)],
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            report=json.loads((path/'report.json').read_text())
            for key in ['training_features_equal','training_targets_equal','scalers_equal',
                        'weights_equal','torch_predictions_equal','numpy_exports_equal','numpy_predictions_equal']:
                self.assertTrue(report[key],key)
            self.assertLessEqual(report['numpy_torch_max_difference_MW'],.002)


if __name__=='__main__':unittest.main()
