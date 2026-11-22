"""Dry-run by default. --execute explicitly enables expensive historical fits."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from common import (HERE,ROOT,DEFAULT_DATA,read_json,digest,experiment_key,build_windows,
                    cache_location,validate_cache,fit_snapshot,predict_week,score_week,frozen_model)
from uncertainty import summarize


def inventory(data,windows,key):
    timestamp=pd.read_csv(data,usecols=['timestamp'],parse_dates=['timestamp']).timestamp
    days=(pd.DatetimeIndex(timestamp)-pd.Timedelta(minutes=30)).normalize()
    price_times=read_json(ROOT/'results/current_c2_build/member_timings.json')
    price_rate=sum(item['seconds'] for item in price_times)/293712
    demand_fit=read_json(ROOT/'results/demand_seed_averaging_5seeds_2015_2024/lgbm_9418_2024_fit.json')
    demand_rate=demand_fit['seconds']/demand_fit['training_rows']
    mlp_rate=read_json(ROOT/'results/current_demand_mlp5_build/completion.json')['seconds']/293756
    rows=[]
    for window in windows.to_dict('records'):
        cutoff=pd.Timestamp(window['training_cutoff'])
        prefix_rows=int((days<=cutoff).sum())
        eval_days=pd.date_range(window['week_start'],window['week_end'],freq='D')
        assert all(int((days==day).sum())==48 for day in eval_days)
        assert prefix_rows>1392 and int((days==cutoff).sum())==48
        cached=validate_cache(cache_location(window,key),key)
        estimated=(prefix_rows-1392)*price_rate+(prefix_rows-1348)*(demand_rate+mlp_rate)
        rows.append(dict(**window,cached_exact_snapshot=cached,training_raw_rows=prefix_rows,
            demand_training_rows_estimate=prefix_rows-1348,price_training_rows_estimate=prefix_rows-1392,
            evaluation_halfhours=336,estimated_fit_seconds=0. if cached else estimated))
    table=pd.DataFrame(rows);missing=int((~table.cached_exact_snapshot).sum())
    seconds=float(table.estimated_fit_seconds.sum())
    production=ROOT/'code/forecast_package/artifacts/manifest.json'
    live_ready=False
    if production.exists():
        manifest=read_json(production)
        live_ready=(str(manifest.get('data_end',''))[:10]==read_json(HERE/'plan.json')['live_target']['training_cutoff']
            and manifest.get('demand_mlp_seeds')==list(range(9417,9422))
            and manifest.get('selection')==read_json(HERE/'recipe.json'))
    report=dict(status='dry_run_no_historical_training_started',experiment_key=key,
        seasonal_weeks=int((table.group=='seasonal').sum()),recent_weeks=int((table.group=='recent').sum()),
        exact_historical_snapshots_cached=len(table)-missing,snapshots_missing=missing,
        new_LightGBM_fits=missing*6,new_MLP_fits=missing*5,
        estimated_sequential_hours=seconds/3600,
        rough_hours_range=[seconds*.75/3600,seconds*1.5/3600],
        estimate_note='Linear scaling of existing local timings; serial snapshots, two model threads. Not a benchmark or guaranteed duration.',
        current_production_package_matches_actual_live_cutoff=live_ready,
        production_package_reused_for_historical_tests=False,
        reuse_note='Exact-cutoff caches only. Biennial models, price-only reports and models trained beyond a test cutoff are not reusable combined snapshots.')
    return table,report


def save_reports(results):
    files=sorted((HERE/'predictions').glob('*.csv'))
    windows=build_windows().set_index('window');records=[]
    for path in files:
        metadata=read_json(path.with_suffix('.json'))
        if metadata['experiment_key']!=results:raise ValueError('Predictions from incompatible experiment')
        window=windows.loc[path.stem].to_dict();window['window']=path.stem
        predictions=pd.read_csv(path,index_col='timestamp',parse_dates=True)
        if digest(path)!=metadata['predictions_sha256']:raise ValueError('Prediction checksum mismatch')
        for row in score_week(predictions):records.append(dict(**window,**row))
    weekly=pd.DataFrame(records)
    if weekly.empty:return
    output=HERE/'results';output.mkdir(exist_ok=True)
    weekly.to_csv(output/'weekly_metrics.csv',index=False)
    summary,paired,sensitivity=summarize(weekly,read_json(HERE/'plan.json')['bootstrap'])
    summary.to_csv(output/'summary.csv',index=False)
    paired.to_csv(output/'paired_improvement.csv',index=False)
    sensitivity.to_csv(output/'recent_pair_block_sensitivity.csv',index=False)
    plot(weekly,summary)


def plot(weekly,summary):
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='live_week_plot_'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=HERE/'figures';destination.mkdir(exist_ok=True)
    for group,table in weekly.groupby('group'):
        fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
        for axis,metric,label in [(axes[0],'demand_MAPE','Demand MAPE (%) · lower is better'),
                                  (axes[1],'price_balanced_accuracy','Price balanced accuracy (%) · higher is better')]:
            for model,colour in [('current','#147D92'),('baseline','#9A6D42')]:
                rows=table.loc[table.model==model].sort_values('week_start')
                axis.plot(pd.to_datetime(rows.week_start),rows[metric],marker='o',color=colour,label=model)
            axis.set_ylabel(label);axis.grid(alpha=.2);axis.legend()
        axes[1].set_xlabel('Historical test week start')
        fig.suptitle(f'{group.capitalize()} live-week simulations · fixed model, daily history updates')
        fig.text(.5,.02,'Training ends 37 days before each week starts. Each point is a seven-day score; group-mean confidence intervals are in summary.csv.',ha='center',fontsize=8)
        fig.autofmt_xdate();fig.tight_layout(rect=(0,.07,1,.95))
        fig.savefig(destination/f'{group}_weekly_performance.png',dpi=180)
        fig.savefig(destination/f'{group}_weekly_performance.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    labels={'demand_MAPE':'Demand MAPE (%) · lower is better','demand_MAE':'Demand MAE (MW) · lower is better',
            'price_accuracy':'Price accuracy (%) · higher is better',
            'price_balanced_accuracy':'Mean weekly balanced accuracy (%) · higher is better'}
    for axis,metric in zip(axes.flat,labels):
        rows=summary.loc[summary.metric==metric].sort_values(['group','model']).reset_index(drop=True)
        for position,row in rows.iterrows():
            colour='#147D92' if row.model=='current' else '#9A6D42'
            if np.isfinite(row.empirical_week_p10):
                axis.plot([row.empirical_week_p10,row.empirical_week_p90],[position]*2,color=colour,lw=8,alpha=.2)
            if np.isfinite(row.mean_CI95_low):
                axis.plot([row.mean_CI95_low,row.mean_CI95_high],[position]*2,color=colour,lw=2)
            axis.scatter([row['mean']],[position],color=colour,s=35,zorder=3)
        axis.set_yticks(range(len(rows)),[f'{row.group}: {row.model}' for row in rows.itertuples()])
        axis.invert_yaxis();axis.set_xlabel(labels[metric]);axis.grid(axis='x',alpha=.2)
    fig.suptitle('Historical weekly variation and confidence in the historical mean')
    fig.text(.5,.025,'Dot: mean  |  Thin line: 95% bootstrap CI of the mean (at least five weeks)  |  Wide pale line: observed weekly 10th–90th percentiles, not a live-week prediction interval.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.95))
    fig.savefig(destination/'summary_intervals.png',dpi=180)
    fig.savefig(destination/'summary_intervals.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Run missing fits; can take many hours')
    parser.add_argument('--group',choices=['all','seasonal','recent'],default='all')
    parser.add_argument('--window',action='append',help='Run only named window(s), e.g. seasonal_2025')
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    args=parser.parse_args()
    windows=build_windows();key=experiment_key(args.data)
    windows.to_csv(HERE/'windows.csv',index=False)
    table,report=inventory(args.data,windows,key)
    table.to_csv(HERE/'inventory.csv',index=False)
    (HERE/'inventory.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not args.execute:
        print('Dry run only. No historical models fitted. Use --execute to opt in.',flush=True);return
    selected=windows
    if args.group!='all':selected=selected.loc[selected.group==args.group]
    if args.window:
        unknown=set(args.window)-set(windows.window)
        if unknown:raise ValueError(f'Unknown windows: {unknown}')
        selected=selected.loc[selected.window.isin(args.window)]
    if selected.empty:raise ValueError('No selected windows')
    frame=frozen_model().load_data(args.data);(HERE/'predictions').mkdir(exist_ok=True)
    started=time.monotonic()
    for window in selected.to_dict('records'):
        print(f'Window {window["window"]}: train through {window["training_cutoff"]}, predict {window["week_start"]}–{window["week_end"]}',flush=True)
        destination=HERE/'predictions'/f'{window["window"]}.csv';metadata=destination.with_suffix('.json')
        if destination.exists() and metadata.exists():
            previous=read_json(metadata)
            if previous['experiment_key']!=key or previous['predictions_sha256']!=digest(destination):
                raise ValueError('Existing predictions incompatible or modified')
        else:
            snapshot=fit_snapshot(frame,window,key)
            predictions=predict_week(frame,window,snapshot,key)
            predictions.to_csv(destination,index_label='timestamp')
            metadata.write_text(json.dumps(dict(experiment_key=key,window=window,
                snapshot=str(snapshot.relative_to(HERE)),predictions_sha256=digest(destination)),indent=2))
        save_reports(key)
        (HERE/'run_status.json').write_text(json.dumps(dict(status='running',last_completed=window['window'],elapsed_seconds=time.monotonic()-started),indent=2))
    (HERE/'run_status.json').write_text(json.dumps(dict(status='selected_windows_complete',windows=list(selected.window),seconds=time.monotonic()-started),indent=2))


if __name__=='__main__':main()
