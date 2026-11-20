import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'forecast_package'))
import model
from c2_price import price_table
from c2_recipe import weather_shape_names
from test_forecasting import synthetic_data


class ProbabilityMember:
    def __init__(self,probabilities): self.probabilities=np.array(probabilities)
    def predict_proba(self,x): return np.column_stack([1-self.probabilities,self.probabilities])


class C2Tests(unittest.TestCase):
    def test_average_probabilities_before_strict_threshold(self):
        bundle=dict(task='price',features=['x'],threshold=.5,
                    members=[ProbabilityMember([.1,.9,.5]),ProbabilityMember([.8,.3,.5])])
        np.testing.assert_array_equal(model.predict_task(bundle,pd.DataFrame({'x':[0,0,0]})),[0,1,0])

    def test_weather_shape_live_matches_training_and_is_causal(self):
        frame=synthetic_data()
        x,y=price_table(frame)
        self.assertEqual(x.shape[1],597)
        self.assertEqual(list(x.columns[-26:]),weather_shape_names())
        days=model.market_days(frame.index); day=days.unique()[-2]
        history=frame.loc[days<day]; weather=frame.loc[days==day,model.WEATHER]
        live=model.forecast_features(history,weather,weather_shape=True).loc[:,x.columns]
        pd.testing.assert_frame_equal(live,x.loc[weather.index])
        changed=frame.copy(); changed.loc[days>=day,model.MARKET]=-99999
        changed.loc[days>day,model.WEATHER]=99999
        altered,_=price_table(changed)
        pd.testing.assert_frame_equal(x.loc[weather.index],altered.loc[weather.index])
        self.assertTrue((pd.Series(1,index=x.index).groupby(model.market_days(x.index)).sum()==48).all())


if __name__=='__main__': unittest.main()
