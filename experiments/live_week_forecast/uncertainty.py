"""Uncertainty over complete weekly scores, never independent half-hours."""
import numpy as np
import pandas as pd

METRICS=['demand_MAPE','demand_MAE','price_accuracy','price_balanced_accuracy']


def bootstrap_indices(n,replicates=5000,seed=9417,block_windows=1):
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(replicates,int(np.ceil(n/block_windows))))
    return ((starts[:,:,None]+np.arange(block_windows))%n).reshape(replicates,-1)[:,:n]


def summarize(weekly,settings):
    summaries=[];comparisons=[];sensitivity=[]
    for group,table in weekly.groupby('group'):
        for metric in METRICS:
            values=table.pivot(index='week_start',columns='model',values=metric).sort_index()
            if values.isna().any().any():raise ValueError('Unpaired current/baseline weeks')
            n=len(values);enough=n>=settings['minimum_weeks_for_CI']
            indices=bootstrap_indices(n,settings['replicates'],settings['seed'])
            for model in ['current','baseline']:
                scores=values[model].to_numpy()
                interval=np.percentile(scores[indices].mean(axis=1),[2.5,97.5]) if enough else [np.nan,np.nan]
                quantiles=np.quantile(scores,settings['empirical_quantiles']) if n>1 else [np.nan,np.nan]
                summaries.append(dict(group=group,model=model,metric=metric,weeks=n,
                    mean=float(scores.mean()),median=float(np.median(scores)),minimum=float(scores.min()),maximum=float(scores.max()),
                    empirical_week_p10=float(quantiles[0]),empirical_week_p90=float(quantiles[1]),
                    mean_CI95_low=float(interval[0]),mean_CI95_high=float(interval[1]),
                    interval_status='approximate historical mean CI' if enough else 'too few weeks for CI'))
            # Positive always means the current model improves over yesterday.
            delta=values.baseline.to_numpy()-values.current.to_numpy()
            if metric.startswith('price_'):delta=-delta
            interval=np.percentile(delta[indices].mean(axis=1),[2.5,97.5]) if enough else [np.nan,np.nan]
            comparisons.append(dict(group=group,metric=metric,weeks=n,mean_improvement=float(delta.mean()),
                paired_CI95_low=float(interval[0]),paired_CI95_high=float(interval[1])))
            if group=='recent':
                blocks=bootstrap_indices(n,settings['replicates'],settings['seed'],settings['recent_sensitivity_block_windows'])
                for model in ['current','baseline']:
                    scores=values[model].to_numpy()
                    interval=np.percentile(scores[blocks].mean(axis=1),[2.5,97.5]) if enough else [np.nan,np.nan]
                    sensitivity.append(dict(group=group,model=model,metric=metric,weeks=n,
                        block_windows=settings['recent_sensitivity_block_windows'],mean=float(scores.mean()),
                        mean_CI95_low=float(interval[0]),mean_CI95_high=float(interval[1])))
                interval=np.percentile(delta[blocks].mean(axis=1),[2.5,97.5]) if enough else [np.nan,np.nan]
                sensitivity.append(dict(group=group,model='paired_improvement',metric=metric,weeks=n,
                    block_windows=settings['recent_sensitivity_block_windows'],mean=float(delta.mean()),
                    mean_CI95_low=float(interval[0]),mean_CI95_high=float(interval[1])))
    return pd.DataFrame(summaries),pd.DataFrame(comparisons),pd.DataFrame(sensitivity)
