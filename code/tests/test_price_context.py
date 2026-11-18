"""Causal price context and unchanged demand inference."""
import tempfile
import unittest
import numpy as np
import pandas as pd
import joblib
from test_forecasting import synthetic_data
from model import (MARKET, WEATHER, forecast_features, training_table, market_days,
                   Forecaster, predict_task, predict_price_probability)
from train import fit_model
from price_context import price_context_frame, extend_price_features, forecast_price_features


class PriceContextTests(unittest.TestCase):
    def test_counts_live_and_future_invariance(self):
        f=synthetic_data()
        history,weather=f.iloc[:-48],f.iloc[-48:][WEATHER]
        x,_=training_table(f)
        for name,n in [('base',0),('geometry',12),('recency',7),('both',19)]:
            context=price_context_frame(f,name)
            self.assertEqual(context.shape[1],n)
            expected=extend_price_features(f,x.loc[weather.index],name)
            live=forecast_price_features(history,weather,forecast_features(history,weather),name)
            pd.testing.assert_frame_equal(expected,live)
            changed=f.copy();changed.loc[weather.index,MARKET]=-999999
            pd.testing.assert_frame_equal(context.loc[weather.index],price_context_frame(changed,name).loc[weather.index])

    def test_zero_spread_negative_prices_and_strict_equality(self):
        f=synthetic_data(); f['nsw_price']=-10.
        c=price_context_frame(f).iloc[-48:]
        self.assertTrue(np.isfinite(c.to_numpy()).all())
        self.assertTrue((c.pc_exceed_28==0).all())
        self.assertTrue((c.pc_negative_fraction==1).all())
        self.assertTrue((c.pc_margin_1==0).all())
        f.iloc[-49,f.columns.get_loc('nsw_price')]=10000
        self.assertTrue(np.isfinite(price_context_frame(f).iloc[-48:].to_numpy()).all())

    def test_serialization_preserves_demand_and_price(self):
        f=synthetic_data(start='2024-12-01')
        history,weather=f.iloc[:-48],f.iloc[-48:][WEATHER]
        x,y=training_table(history)
        config=dict(family='boosted',leaves=7,trees=5,learning_rate=.04)
        demand=fit_model('demand',config,x,y)
        px=extend_price_features(history,x,'both')
        price=fit_model('price',dict(config,price_feature_set='both',holiday_context=True),px,y)
        base=forecast_features(history,weather)
        live=forecast_price_features(history,weather,base,'both')
        expected=predict_price_probability(price,live)>.5
        with tempfile.TemporaryDirectory() as path:
            joblib.dump(dict(demand=demand,price=price),path+'/models.joblib')
            p=Forecaster(path).predict(history,weather)
        np.testing.assert_array_equal(p.nsw_demand_mw,predict_task(demand,base))
        np.testing.assert_array_equal(p.nsw_price_up,expected)

    def test_older_features_do_not_depend_on_extension(self):
        short=synthetic_data(days=75);long=synthetic_data(days=90)
        pd.testing.assert_frame_equal(price_context_frame(short),price_context_frame(long).loc[short.index])

if __name__=='__main__': unittest.main()

class PricePolicyTests(unittest.TestCase):
    def test_recency_weights_balance_effective_classes_once(self):
        from unittest.mock import patch
        from lightgbm import LGBMClassifier
        f=synthetic_data(days=130)
        x,y=training_table(f)
        config=dict(family='boosted',leaves=7,trees=2,learning_rate=.04,half_life_years=2)
        with patch.object(LGBMClassifier,'fit',autospec=True) as fitted:
            bundle=fit_model('price',config,x,y)
        weights=fitted.call_args.kwargs['sample_weight']
        self.assertAlmostEqual(weights.mean(),1.)
        self.assertAlmostEqual(weights[y.price.to_numpy()==0].sum(),weights[y.price.to_numpy()==1].sum())
        self.assertIsNone(bundle['estimator'].get_params()['class_weight'])

    def test_xgboost_price_probability_and_training_cutoff(self):
        f=synthetic_data()
        x,y=training_table(f.iloc[:-48])
        config=dict(family='xgboost',trees=2,max_depth=2,learning_rate=.04)
        bundle=fit_model('price',config,x,y)
        live=forecast_features(f.iloc[:-48],f.iloc[-48:][WEATHER])
        p=predict_price_probability(bundle,live)
        self.assertTrue(((p>=0)&(p<=1)).all())
        with self.assertRaises(ValueError): predict_price_probability(bundle,x.iloc[-48:])

class PriceRunnerTests(unittest.TestCase):
    def test_short_backtest_keeps_frozen_output_and_past_training(self):
        from run_price_experiments import backtest, threshold, paired
        f=synthetic_data(days=85,start='2014-11-01')
        x,y=training_table(f)
        end=market_days(x.index).max()
        config=dict(family='boosted',leaves=7,trees=2,learning_rate=.04,threshold=.57)
        p,fits=backtest(config,x,y,end)
        self.assertEqual(len(fits),1)
        self.assertLess(pd.Timestamp(fits[0]['training_end']),pd.Timestamp(fits[0]['effective_day']))
        np.testing.assert_array_equal(p.prediction,p.frozen_prediction)
        self.assertEqual(paired(p,p,draws=20)['delta_BA_pp'],0.)
        self.assertTrue(.05<=threshold([0,1],np.array([.2,.8]))<=.95)
