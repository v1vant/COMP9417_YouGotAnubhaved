"""Optional, previous-day-only price-spike context for controlled experiments."""
import numpy as np
import pandas as pd


def longest_run(row):
    changes = np.flatnonzero(np.diff(np.r_[False,row,False].astype(int)))
    return int(np.max(changes[1::2]-changes[::2])) if len(changes) else 0


def spike_features(frame):
    days = (frame.index-pd.Timedelta(minutes=30)).normalize()
    slots = np.tile(np.arange(48),len(frame)//48)
    profiles = pd.DataFrame({"day":days,"slot":slots,"price":frame.nsw_price.to_numpy()}).pivot(
        index="day",columns="slot",values="price")
    threshold = frame.nsw_price.rolling(48*30,min_periods=48*30).quantile(.99).groupby(days).last().shift(1)
    exceed = profiles.gt(threshold,axis=0)
    excess = profiles.sub(threshold,axis=0).clip(lower=0)
    mean = profiles.mean(axis=1)
    ordered = np.sort(profiles.to_numpy(),axis=1)
    nonspike_mean = profiles.where(~exceed).mean(axis=1)
    trimmed = pd.DataFrame({
        "mean_without_top1":ordered[:,:-1].mean(axis=1),
        "mean_without_top3":ordered[:,:-3].mean(axis=1),
        "mean_without_spikes":nonspike_mean.fillna(mean),
        "no_nonspike_intervals":nonspike_mean.isna().astype(float),
    },index=profiles.index)
    trimmed["top1_mean_distortion"] = mean-trimmed.mean_without_top1
    trimmed["top3_mean_distortion"] = mean-trimmed.mean_without_top3
    trimmed["spike_mean_distortion"] = mean-trimmed.mean_without_spikes
    count = exceed.sum(axis=1)
    last_date = pd.Series(profiles.index.where(count.gt(0)),index=profiles.index).ffill()
    age = (pd.Series(profiles.index,index=profiles.index)-last_date).dt.days
    relative = pd.DataFrame({
        "previous_30day_q99":threshold,
        "mean_minus_q99":mean-threshold,
        "count":count,
        "largest_excess":excess.max(axis=1),
        "mean_excess":excess.mean(axis=1),
        "last_spike_slot":np.where(exceed.to_numpy().any(axis=1),
                                   np.where(exceed.to_numpy(),np.arange(48),-1).max(axis=1),-1),
        "longest_run":np.array([longest_run(row) for row in exceed.to_numpy()]),
        "spike_days_last7":count.gt(0).rolling(7,min_periods=7).sum(),
        "days_since_spike":age.fillna(31).clip(upper=30),
        "no_spike_last30":(age.isna()|age.gt(30)).astype(float),
    },index=profiles.index)
    for block in range(4):
        relative[f"count_hours_{block*6:02d}_{(block+1)*6:02d}"] = exceed.iloc[:,block*12:(block+1)*12].sum(axis=1)
    absolute = pd.DataFrame({"count_above_300":profiles.gt(300).sum(axis=1),
                             "count_above_1000":profiles.gt(1000).sum(axis=1)},index=profiles.index)
    lookup = pd.Series(days,index=frame.index)
    output, groups = {}, {}
    for group, table in (("trimmed",trimmed),("relative",relative),("absolute",absolute)):
        groups[group] = []
        for name in table:
            column = f"price_spike_{group}_{name}"
            output[column] = lookup.map(table[name].shift(1))
            groups[group].append(column)
    return pd.DataFrame(output,index=frame.index,dtype=np.float32),groups
