"""Strong-baseline feature, threshold and ensemble research; no automatic deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research import (ROOT, BASE, fit, refresh, metrics, threshold, paired, reports,
                      backtest, load_data, training_table, market_days, predict_price_probability)
from price_context import extend_price_features


def table(frame, compact=False, context=False):
    x,y=training_table(frame)
    # Exactly match harness Series.mean(), including strict floating-point ties.
    means=frame.nsw_price.groupby(market_days(frame.index)).agg(lambda s:s.mean()).shift(1)
    y['price']=(frame.loc[x.index,'nsw_price'].to_numpy()>means.reindex(market_days(x.index)).to_numpy()).astype(int)
    if compact:
        x=x.drop(columns=[c for c in x if any(s in c for s in ('_profile_','_before_','_ramp_','_change_from_open')) or c=='nsw_price_weekly_margin'])
        assert x.shape[1]==240
    if context:x=extend_price_features(frame,x,'both')
    counts=pd.Series(1,index=x.index).groupby(market_days(x.index)).sum()
    keep=market_days(x.index).isin(counts[counts==48].index)
    return x.loc[keep],y.loc[keep]


def calibration(config,x,y,cutoff):
    start=cutoff-pd.Timedelta(days=365);middle=start+pd.Timedelta(days=182)
    days=market_days(x.index); parts=[]
    for a,b in [(start,middle),(middle,cutoff)]:
        m=fit(config,x,y,a)
        mask=(days>=a)&(days<b)
        parts.append(pd.Series(predict_price_probability(m,x.loc[mask]),index=x.index[mask]))
    return pd.concat(parts)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--seed',type=int,default=9417)
    ap.add_argument('--hours',type=float,default=24)
    args=ap.parse_args()
    if args.hours<=0:ap.error('--hours must be positive')
    if args.output.exists() and any(args.output.iterdir()):ap.error('Use an empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();deadline=started+args.hours*3600
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    frame=load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv')
    frame=frame.loc[market_days(frame.index)<=pd.Timestamp('2024-12-31')]
    config=dict(BASE,trees=20 if args.quick else 600,seed=args.seed)
    status=dict(state='running',seed=args.seed,quick=args.quick,source_commit='f92a38e',sources=hashes,completed=[])
    def save(): (args.output/'status.json').write_text(json.dumps(status,indent=2))
    save(); results={};validation={};records=[];deltas=[];cal={};configs={}
    for name,compact,context,adaptive in [('P0',False,False,False),('P1',False,True,False),
                                         ('P2',False,False,True),('P3',False,True,True),
                                         ('compact',True,True,True)]:
        if time.monotonic()>deadline:break
        status['current']=name;save()
        x,y=table(frame,compact,context); days=market_days(x.index)
        c=dict(config,price_feature_set='both' if context else 'base',compact=compact)
        m=fit(c,x,y,pd.Timestamp('2014-01-01')); mask=days.year==2014
        vp=pd.Series(predict_price_probability(m,x.loc[mask]),index=x.index[mask])
        validation[name]=vp
        pd.DataFrame(dict(actual=y.loc[mask,'price'],probability=vp)).to_parquet(args.output/f'{name}_validation.parquet')
        p,fits=backtest(c,x,y,pd.Timestamp('2024-12-31'),adaptive=adaptive)
        p.to_parquet(args.output/f'{name}_predictions.parquet');results[name]=p
        pd.DataFrame(fits).to_csv(args.output/f'{name}_fits.csv',index=False)
        records.extend(reports(name,p)); configs[name]=dict(config=c,refresh=adaptive)
        dev=market_days(p.index).year<=2022
        deltas.append(dict(candidate=name,**paired(results['P0'].loc[dev],p.loc[dev])))
        # Cache past-only component scores at each replacement for ensemble thresholds.
        if name in ['P0','compact']:
            parts=[]
            for year in [2015,2016,2018,2020,2022,2024]:
                cp=calibration(c,x,y,pd.Timestamp(year=year,month=1,day=1))
                parts.append(pd.DataFrame(dict(probability=cp,actual=y.loc[cp.index,'price'],effective_year=year)))
            cal[name]=pd.concat(parts)
            cal[name].to_parquet(args.output/f'{name}_calibration.parquet')
        status['completed'].append(name);save()
        pd.DataFrame(records).to_csv(args.output/'metrics.csv',index=False)
        pd.DataFrame(deltas).to_csv(args.output/'paired_deltas.csv',index=False)
        print(name,deltas[-1],flush=True)
    if 'compact' in results:
        # Weight selected only on 2014, before inspecting the simulation scores.
        yv=pd.read_parquet(args.output/'P0_validation.parquet').actual
        options=[]
        for w in [0.,.25,.5,.75,1.]:
            vp=w*validation['P0']+(1-w)*validation['compact'];t=threshold(yv,vp.to_numpy())
            options.append((metrics(yv,vp>t)['balanced_accuracy'],w,t))
        _,w,t=max(options,key=lambda z:(z[0],z[1]))
        p=results['P0'].copy(); p['probability']=w*p.probability+(1-w)*results['compact'].probability
        for year,a in cal['P0'].groupby('effective_year'):
            b=cal['compact'];b=b[b.effective_year==year]
            if not a.index.equals(b.index):raise ValueError('Mismatched calibration dates')
            t=threshold(a.actual,(w*a.probability+(1-w)*b.probability).to_numpy(),t)
            d=market_days(p.index); stop={2015:2016,2016:2018,2018:2020,2020:2022,2022:2024,2024:2025}[year]
            mask=(d.year>=year)&(d.year<stop); p.loc[mask,'threshold']=t
        p['prediction']=(p.probability>p.threshold).astype(int)
        p.to_parquet(args.output/'ensemble_predictions.parquet');records.extend(reports('ensemble',p));results['ensemble']=p
        configs['ensemble']=dict(weight=w,components=['P0','compact'],refresh=True)
    # Stage C is conditional on the gain remaining below one BA point.
    dev=market_days(results['P0'].index).year<=2022
    gain=max(metrics(p.loc[dev,'actual'],p.loc[dev,'prediction'])['balanced_accuracy'] for p in results.values())-metrics(results['P0'].loc[dev,'actual'],results['P0'].loc[dev,'prediction'])['balanced_accuracy']
    if gain<1. and time.monotonic()<deadline and not args.quick:
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            status['stacking']='blocked: torch is unavailable in the active environment'
        else:
            from stack import generate
            status['current']='demand_stacking';save()
            labels=pd.read_parquet(args.output/'P0_validation.parquet').actual
            use_context=metrics(labels,validation['P1']>.57)['balanced_accuracy']>metrics(labels,validation['P0']>.57)['balanced_accuracy']
            full,truth=table(frame,context=use_context)
            extra=generate(frame,full,truth,args.output/'demand_oof')
            # Both arms exclude 2010; this is a matched-date comparison.
            bx=full.loc[extra.index];by=truth.loc[extra.index]
            for label,tx in [('stack_control',bx),('stacked',pd.concat([bx,extra],axis=1))]:
                p,fits=backtest(dict(config,price_feature_set='both' if use_context else 'base'),tx,by,pd.Timestamp('2024-12-31'),adaptive=True)
                p.to_parquet(args.output/f'{label}_predictions.parquet')
                pd.DataFrame(fits).to_csv(args.output/f'{label}_fits.csv',index=False)
                results[label]=p;records.extend(reports(label,p))
                configs[label]=dict(config=dict(config,price_feature_set='both' if use_context else 'base'),refresh=True,stacked=label=='stacked',training_start='2011-01-01')
            d=market_days(results['stacked'].index).year<=2022
            (args.output/'stack_paired_delta.json').write_text(json.dumps(paired(results['stack_control'].loc[d],results['stacked'].loc[d]),indent=2))
            status['stacking']='completed'
    pd.DataFrame(records).to_csv(args.output/'metrics.csv',index=False)
    # A single combined summary, always against the 571-feature control.
    summary=[]
    for name,p in results.items():
        dev=market_days(p.index).year<=2022;conf=~dev
        delta=paired(results['P0'].loc[dev],p.loc[dev]);wins=0
        for year in range(2015,2023):
            m=market_days(p.index).year==year
            wins+=metrics(p.loc[m,'actual'],p.loc[m,'prediction'])['balanced_accuracy']>metrics(results['P0'].loc[m,'actual'],results['P0'].loc[m,'prediction'])['balanced_accuracy']
        acc=metrics(p.loc[dev,'actual'],p.loc[dev,'prediction'])['accuracy']-metrics(results['P0'].loc[dev,'actual'],results['P0'].loc[dev,'prediction'])['accuracy']
        confirmation=metrics(p.loc[conf,'actual'],p.loc[conf,'prediction'])['balanced_accuracy']-metrics(results['P0'].loc[conf,'actual'],results['P0'].loc[conf,'prediction'])['balanced_accuracy']
        summary.append(dict(candidate=name,**delta,positive_years=int(wins),accuracy_delta_pp=acc,confirmation_delta_pp=confirmation,
                            eligible=bool(delta['delta_BA_pp']>=.25 and delta['low']>0 and wins>=6 and acc>=-.25 and confirmation>0)))
    pd.DataFrame(summary).to_csv(args.output/'summary.csv',index=False)
    (args.output/'configurations.json').write_text(json.dumps(configs,indent=2))
    status.update(state='completed' if len(status['completed'])==5 else 'budget_exhausted',seconds=time.monotonic()-started)
    if status.get('stacking','').startswith('blocked:'):
        status['state']='completed_with_blocked_stacking'
    save()

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
