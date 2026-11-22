"""Frozen C2 price recipe; no change to the demand feature table or model."""
import time
import numpy as np
import pandas as pd
from model import market_days, training_table
from c2_recipe import fit_price, weather_shape_names


def price_table(frame):
    """Build complete-day C2 features and strict previous-day-mean targets.

    Weather-shape columns are appended in a stable order so the saved feature
    schema agrees with inference. Partial market days are excluded.
    """
    # Base features plus the weather-shape columns.
    x,y=training_table(frame,weather_shape=True)

    # Move weather-shape columns to the end so the column order is stable.
    shape=weather_shape_names()
    x=x.loc[:,[c for c in x if c not in shape]+shape]

    # Previous market day's mean price is the up/down threshold.
    means=frame.nsw_price.groupby(market_days(frame.index)).agg(lambda s:s.mean()).shift(1)
    y=y.copy()
    y['price']=(frame.loc[x.index,'nsw_price'].to_numpy()>means.reindex(market_days(x.index)).to_numpy()).astype(int)

    # Keep only complete market days (48 half-hour intervals).
    counts=pd.Series(1,index=x.index).groupby(market_days(x.index)).sum()
    keep=market_days(x.index).isin(counts[counts==48].index)

    # Sanity check on the expected feature count.
    assert x.shape[1]==597
    return x.loc[keep],y.loc[keep]


def fit_c2(x,y,config):
    """Fit one classifier per configured seed and return an ensemble bundle."""
    # Validate the input schema and the frozen configuration.
    if x.shape[1]!=597 or not set(weather_shape_names()).issubset(x.columns):
        raise ValueError('C2 requires its 597-feature price table, including weather shape')
    if config.get('half_life_years') or config.get('near_threshold_scale'):
        raise ValueError('Frozen C2 uses neither recency nor margin weighting')

    # Train one LightGBM member per seed.
    members=[]
    for seed in config['seeds']:
        started=time.monotonic()
        members.append(fit_price(x,y.price,seed,config['params']))
        print(f'C2 seed {seed} completed in {time.monotonic()-started:.1f}s',flush=True)

    # Bundle the members with the metadata needed at inference time.
    days=market_days(x.index)
    return dict(task='price',config=config,features=list(x),estimator=members[0],
                members=members,threshold=float(config['threshold']),
                training_start=str(days.min()),training_end=str(days.max()))
