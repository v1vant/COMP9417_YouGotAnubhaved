"""Refresh demand day-type diagnostics from saved chronological forecasts; no fits."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/forecast_package'))
from model import load_data,market_days


def main():
    output=ROOT/'results/day_types_current_mlp5_2015_2024'
    output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/demand_seed_averaging_5seeds_2015_2024/predictions.csv'
    forecasts=pd.read_csv(source,index_col='timestamp',parse_dates=True,
        usecols=['timestamp','actual','current','bagged_MLP_only_5'])
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2025-01-01 00:00:00']
    days=market_days(forecasts.index);unique_days=days.unique()
    assert len(unique_days)==3653 and unique_days.min()==pd.Timestamp('2015-01-01')
    assert unique_days.max()==pd.Timestamp('2024-12-31')
    np.testing.assert_array_equal(forecasts.actual,frame.loc[forecasts.index,'nsw_demand'])
    truth=forecasts.actual.to_numpy().reshape(-1,48)
    previous=forecasts.current.to_numpy().reshape(-1,48)
    current=forecasts.bagged_MLP_only_5.to_numpy().reshape(-1,48)
    profiles=truth/truth.mean(axis=1,keepdims=True)
    clustering=joblib.load(ROOT/'results/day_type_models.joblib')
    assert clustering['reference']=='2010-2013' and clustering['normalize'] and clustering['selected_clusters']==2
    pca=clustering['pca'];km=clustering['clustering']
    embedding=pca.transform(profiles);labels=km.predict(embedding)
    saved=pd.read_csv(ROOT/'results/day_types.csv',index_col='day',parse_dates=True).loc[unique_days]
    np.testing.assert_array_equal(labels,saved.cluster)
    reference_profiles=pca.inverse_transform(km.cluster_centers_)
    actual_profiles=np.array([profiles[labels==k].mean(axis=0) for k in range(2)])
    table=saved.copy()
    table['previous_current_MAPE']=100*np.mean(np.abs(truth-previous)/truth,axis=1)
    table['current_MLP5_MAPE']=100*np.mean(np.abs(truth-current)/truth,axis=1)
    table['previous_current_MAE']=np.mean(np.abs(truth-previous),axis=1)
    table['current_MLP5_MAE']=np.mean(np.abs(truth-current),axis=1)
    table['MAPE_reduction_pp']=table.previous_current_MAPE-table.current_MLP5_MAPE
    table['weekend']=table.index.dayofweek>=5
    summary=table.groupby('cluster').agg(days=('cluster','size'),
        previous_MAPE=('previous_current_MAPE','mean'),current_MAPE=('current_MLP5_MAPE','mean'),
        previous_MAE=('previous_current_MAE','mean'),current_MAE=('current_MLP5_MAE','mean'),
        MAPE_reduction_pp=('MAPE_reduction_pp','mean'),weekend_share=('weekend','mean'))
    summary['relative_MAPE_reduction_percent']=100*summary.MAPE_reduction_pp/summary.previous_MAPE
    # Resample whole weekly blocks, then condition on type within each draw.
    rng=np.random.default_rng(9417);n=len(table);samples={k:[] for k in range(2)}
    delta=table.MAPE_reduction_pp.to_numpy()
    for _ in range(20):
        starts=rng.integers(0,n,size=(250,int(np.ceil(n/7))))
        indices=((starts[:,:,None]+np.arange(7))%n).reshape(250,-1)[:,:n]
        for k in range(2):
            mask=labels[indices]==k
            samples[k].extend(np.divide((delta[indices]*mask).sum(axis=1),mask.sum(axis=1)))
    for k in range(2):
        summary.loc[k,['gain_CI95_low_pp','gain_CI95_high_pp']]=np.percentile(samples[k],[2.5,97.5])
    shares=pd.crosstab(table.year,table.cluster,normalize='index')
    counts=pd.crosstab(table.year,table.cluster)
    yearly=table.groupby(['year','cluster']).agg(days=('cluster','size'),
        previous_MAPE=('previous_current_MAPE','mean'),current_MAPE=('current_MLP5_MAPE','mean')).reset_index()
    table.to_csv(output/'day_types_and_current_errors.csv',index_label='day')
    summary.to_csv(output/'summary_by_type.csv')
    shares.to_csv(output/'yearly_shares.csv');counts.to_csv(output/'yearly_counts.csv')
    yearly.to_csv(output/'yearly_errors_by_type.csv',index=False)
    pd.DataFrame({'slot':np.arange(48),'interval_start_hour':np.arange(48)/2,
        'type_0_reference':reference_profiles[0],'type_1_reference':reference_profiles[1],
        'type_0_2015_2024':actual_profiles[0],'type_1_2015_2024':actual_profiles[1]}).to_csv(output/'profiles.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='day_type_plot_',dir='/private/tmp'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#344054','text.color':'#182230','xtick.color':'#475467','ytick.color':'#475467',
        'axes.edgecolor':'#D0D5DD','legend.frameon':False,'savefig.facecolor':'white'})
    colours=['#147D92','#8A58A0']
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    fig.subplots_adjust(left=.075,right=.97,top=.855,bottom=.145,hspace=.40,wspace=.25)
    fig.suptitle('Current demand ensemble: day-type analysis',fontsize=20,fontweight='bold',y=.975)
    fig.text(.5,.925,'2015–2024 out-of-sample forecasts  |  Current: 50% one LightGBM + 50% mean of five MLPs',ha='center',fontsize=11,color='#475467')
    ax=axes[0,0];hours=np.arange(48)/2
    for k in range(2):
        ax.plot(hours,actual_profiles[k],color=colours[k],lw=2.5,label=f'Type {k} · {int(summary.loc[k,"days"]):,} days')
        ax.plot(hours,reference_profiles[k],color=colours[k],lw=1.4,ls='--',alpha=.65)
    ax.axhline(1,color='#98A2B3',lw=.8,zorder=0)
    ax.set(title='A  |  Demand shape within each type',xlabel='Interval start (AEST)',ylabel='Demand / daily mean',xlim=(0,24),ylim=(.68,1.26))
    ax.set_xticks([0,6,12,18,24],['00:00','06:00','12:00','18:00','24:00'])
    ax.legend(loc='upper left',fontsize=9)
    ax.text(.02,.025,'Solid: 2015–2024 mean   Dashed: fixed 2010–2013 centroid',transform=ax.transAxes,fontsize=8.5,color='#667085')
    ax=axes[0,1]
    ax.stackplot(shares.index,*[100*shares[k] for k in range(2)],colors=colours,alpha=.85,labels=['Type 0','Type 1'])
    ax.set(title='B  |  Share of days assigned to each type',xlabel='Year',ylabel='Share of days (%)',ylim=(0,100),xlim=(2015,2024))
    ax.set_xticks([2015,2017,2019,2021,2023,2024]);ax.legend(loc='lower left',ncol=2,facecolor='white')
    ax=axes[1,0];positions=np.arange(2);width=.33
    bars0=ax.bar(positions-width/2,summary.previous_MAPE,width,color='#ABB3BE',label='Previous: one MLP')
    bars1=ax.bar(positions+width/2,summary.current_MAPE,width,color='#D47735',label='Current: five MLPs')
    ax.bar_label(bars0,fmt='%.3f%%',padding=4,fontsize=10)
    ax.bar_label(bars1,fmt='%.3f%%',padding=4,fontsize=10)
    ax.set_xticks(positions,['Type 0','Type 1'])
    ax.set(title='C  |  Demand error by day type',ylabel='MAPE (%) · lower is better',ylim=(0,float(summary.previous_MAPE.max())*1.25))
    ax.legend(loc='upper center',ncol=2,fontsize=9)
    ax=axes[1,1]
    for k in range(2):
        rows=yearly.loc[yearly.cluster==k]
        ax.plot(rows.year,rows.current_MAPE,color=colours[k],lw=2.2,marker='o',ms=4,label=f'Type {k}')
    last_type1=yearly.loc[(yearly.year==2024)&(yearly.cluster==1)].iloc[0]
    ax.annotate(f'{int(last_type1.days)} days in 2024',xy=(2024,last_type1.current_MAPE),
                xytext=(-8,-15),textcoords='offset points',ha='right',fontsize=8.5,color=colours[1])
    ax.set(title='D  |  Current model error through time',xlabel='Year',ylabel='MAPE (%) · lower is better')
    ax.set_xticks([2015,2017,2019,2021,2023,2024]);ax.legend(loc='upper left')
    for ax in axes.flat:ax.grid(axis='y',color='#EAECF0',lw=.7);ax.set_axisbelow(True)
    previous_score=table.previous_current_MAPE.mean();current_score=table.current_MLP5_MAPE.mean()
    fig.text(.075,.078,f'Overall MAPE: {previous_score:.4f}% → {current_score:.4f}%   |   Same LightGBM, same 50/50 weights; only MLP seed averaging changed.',fontsize=10,fontweight='bold')
    fig.text(.075,.042,'Types use actual completed-day demand shapes: retrospective diagnostics, not prediction inputs. PCA/K-means fixed on 2010–2013; no refitting or holdout data.',fontsize=9,color='#667085')
    fig.savefig(output/'current_demand_day_types.png',dpi=180)
    fig.savefig(output/'current_demand_day_types.pdf')
    plt.close(fig)
    (output/'provenance.json').write_text(json.dumps(dict(current_commit='783e503',
        predictions_source=str(source.relative_to(ROOT)),predictions_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cluster_source='results/day_type_models.joblib',cluster_reference='2010-2013',
        clusters_refitted=False,evaluation_start='2015-01-01',evaluation_end='2024-12-31',
        holdout_used=False,price_analysis_included=False,rows=len(forecasts),days=len(table)),indent=2))
    print(summary.to_string(),flush=True)
    print('Plot saved:',output/'current_demand_day_types.png',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
