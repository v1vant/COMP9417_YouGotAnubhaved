"""Frozen C2 price recipe; no change to the demand feature table or model."""
import time
import numpy as np
import pandas as pd
from model import market_days, training_table
from c2_recipe import fit_price, weather_shape_names


def price_table(frame):
    x,y=training_table(frame,weather_shape=True)
    shape=weather_shape_names()
    x=x.loc[:,[c for c in x if c not in shape]+shape]
    means=frame.nsw_price.groupby(market_days(frame.index)).agg(lambda s:s.mean()).shift(1)
    y=y.copy()
    y['price']=(frame.loc[x.index,'nsw_price'].to_numpy()>means.reindex(market_days(x.index)).to_numpy()).astype(int)
    counts=pd.Series(1,index=x.index).groupby(market_days(x.index)).sum()
    keep=market_days(x.index).isin(counts[counts==48].index)
    assert x.shape[1]==597
    return x.loc[keep],y.loc[keep]


def fit_c2(x,y,config):
    if x.shape[1]!=597 or not set(weather_shape_names()).issubset(x.columns):
        raise ValueError('C2 requires its 597-feature price table, including weather shape')
    if config.get('half_life_years') or config.get('near_threshold_scale'):
        raise ValueError('Frozen C2 uses neither recency nor margin weighting')
    members=[]
    for seed in config['seeds']:
        started=time.monotonic()
        members.append(fit_price(x,y.price,seed,config['params']))
        print(f'C2 seed {seed} completed in {time.monotonic()-started:.1f}s',flush=True)
    days=market_days(x.index)
    return dict(task='price',config=config,features=list(x),estimator=members[0],
                members=members,threshold=float(config['threshold']),
                training_start=str(days.min()),training_end=str(days.max()))
