import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path.cwd();out=ROOT/'results/spike_exploration_2015_2020';out.mkdir(exist_ok=False)
f=pd.read_csv('data/nem_weather_halfhourly_2010_2026.csv',usecols=['timestamp','nsw_price'],index_col='timestamp',parse_dates=True).loc[:'2021-01-01 00:00:00']
days=(f.index-pd.Timedelta(minutes=30)).normalize();lookup=pd.Series(days,index=f.index);daily=f.nsw_price.groupby(days).agg(['mean','median','max']);daily['trimmed_mean']=(48*daily['mean']-daily['max'])/47;daily['max_mean_effect']=daily['mean']-daily['trimmed_mean']
mask=(days>=pd.Timestamp('2015-01-01'))&(days<=pd.Timestamp('2020-12-31'));idx=f.index[mask];d=days[mask];price=f.nsw_price.loc[idx].to_numpy();threshold=lookup.map(daily['mean'].shift(1)).loc[idx].to_numpy();trimthreshold=lookup.map(daily.trimmed_mean.shift(1)).loc[idx].to_numpy();y=(price>threshold).astype(int)
p=np.load('results/feature_comparison_2015_2024/current_price.npy').ravel()[:len(y)];assert len(p)==len(y)
base=(f.nsw_price.shift(48).loc[idx].to_numpy()>threshold).astype(int)
recent_q=f.nsw_price.rolling(48*30,min_periods=48*30).quantile(.99).groupby(days).last().shift(1)
def score(m, prediction=p):
 yy=y[m];pp=prediction[m];neg=yy==0;pos=yy==1
 rec0=np.mean(pp[neg]==0)*100 if neg.any() else np.nan;rec1=np.mean(pp[pos]==1)*100 if pos.any() else np.nan
 return {'intervals':int(m.sum()),'days':len(pd.DatetimeIndex(d[m]).unique()),'positives':int(pos.sum()),'positive_percent':float(pos.mean()*100),'accuracy':float(np.mean(yy==pp)*100),'balanced_accuracy':float(np.nanmean([rec0,rec1])),'recall_1':float(rec1),'false_negatives':int(np.sum(pos&(pp==0))),'false_positives':int(np.sum(neg&(pp==1)))}
frequency=[];categories=[];effects=[];timing=[];episodes=[];impacts=[];yearly=[];persistence=[]
for name,sp in [('above_300',f.nsw_price>300),('above_1000',f.nsw_price>1000),('above_previous_30day_q99',f.nsw_price>lookup.map(recent_q))]:
 count=sp.groupby(days).sum();day_spike=count.gt(0);category=pd.Series('none',index=count.index);category[count==1]='isolated_1';category[count.between(2,3)]='short_2_3';category[count>=4]='sustained_4plus'
 current=lookup.loc[idx].map(day_spike).to_numpy(dtype=bool);after=lookup.loc[idx].map(day_spike.shift(1,fill_value=False)).to_numpy(dtype=bool);previous_type=lookup.loc[idx].map(category.shift(1)).to_numpy()
 frequency.append({'definition':name,'spike_intervals':int(sp.loc[idx].sum()),'interval_percent':float(sp.loc[idx].mean()*100),'spike_days':int(current.sum()/48),'spike_day_percent':float(current.mean()*100),'isolated_days':int(((count==1)&(count.index.year>=2015)&(count.index.year<=2020)).sum()),'sustained_days':int(((count>=4)&(count.index.year>=2015)&(count.index.year<=2020)).sum())})
 valid_day=(day_spike.index>=pd.Timestamp('2015-01-01'))&(day_spike.index<pd.Timestamp('2020-12-31'));today=day_spike.loc[valid_day];tomorrow=day_spike.shift(-1).loc[valid_day].astype(bool)
 persistence.append({'definition':name,'next_day_spike_percent_after_spike':float(tomorrow[today].mean()*100),'next_day_spike_percent_after_nonspike':float(tomorrow[~today].mean()*100),'spike_source_days':int(today.sum())})
 eligible=sp.loc[idx].to_numpy();starts=eligible&~np.r_[False,eligible[:-1]];runs=np.cumsum(starts);lengths=pd.Series(eligible.astype(int)).groupby(runs).sum();lengths=lengths[lengths>0]
 episodes.append({'definition':name,'episodes':len(lengths),'median_intervals':float(lengths.median()),'p90_intervals':float(lengths.quantile(.9)),'max_intervals':int(lengths.max())})
 for group,m in [('all',np.ones(len(y),dtype=bool)),('ordinary_no_spike_today_or_yesterday',~current&~after),('spike_day',current),('after_any_spike',after)]+[(f'after_{c}',previous_type==c) for c in ('isolated_1','short_2_3','sustained_4plus')]:
  if not m.any():continue
  record={'definition':name,'group':group,**score(m)};baseline=score(m,base);record.update(baseline_accuracy=baseline['accuracy'],baseline_balanced_accuracy=baseline['balanced_accuracy']);categories.append(record)
  if group.startswith('after'):
   source_effect=lookup.loc[idx].map(daily.max_mean_effect.shift(1)).to_numpy()[m].reshape(-1,48)[:,0]
   changed=((price>trimthreshold)&(price<=threshold)&m)
   effects.append({'definition':name,'group':group,'days':len(source_effect),'median_mean_increase_due_to_largest_interval':float(np.median(source_effect)),'p90_mean_increase_due_to_largest_interval':float(np.quantile(source_effect,.9)),'mean_increase_due_to_largest_interval':float(np.mean(source_effect)),'label_changes_if_largest_removed':int(changed.sum()),'label_change_percent':float(changed.sum()/m.sum()*100)})
 for year in range(2015,2021):
  yr=d.year==year;yearly.append({'definition':name,'year':year,'spike_day_percent':float(current[yr].mean()*100),'overall_balanced_accuracy':score(yr)['balanced_accuracy'],'ordinary_balanced_accuracy':score(yr&~current&~after)['balanced_accuracy']})
 for block in range(4):
  hour=(idx-pd.Timedelta(minutes=30)).hour;m=after&(hour>=block*6)&(hour<(block+1)*6)
  if m.any():timing.append({'definition':name,'time_block':f'{block*6:02d}–{(block+1)*6:02d}',**score(m)})
 fn=after&(y==1)&(p==0);fp=after&(y==0)&(p==1)
 impacts.append({'definition':name,'after_days':int(after.sum()/48),'missed_positives_after_spikes':int(fn.sum()),'share_all_missed_positives_percent':float(fn.sum()/np.sum((y==1)&(p==0))*100),'FN_only_oracle_global_BA_gain_pp':float(fn.sum()/np.sum(y==1)*50),'all_error_oracle_global_BA_gain_pp':float((fn.sum()/np.sum(y==1)+fp.sum()/np.sum(y==0))*50),'missed_positive_margin_median':float(np.median((price-threshold)[fn])) if fn.any() else None,'missed_positive_within_10_dollars_percent':float(np.mean((price-threshold)[fn]<=10)*100) if fn.any() else None})
for file,data in [('frequency',frequency),('persistence',persistence),('episodes',episodes),('afterday_metrics',categories),('threshold_distortion',effects),('afterday_timing',timing),('yearly',yearly),('global_error_impact',impacts)]:pd.DataFrame(data).to_csv(out/f'{file}.csv',index=False)
metadata={'analysis_start':'2015-01-01','analysis_end':'2020-12-31','rows':len(y),'reference_predictions':'results/feature_comparison_2015_2024/current_price.npy; prefix only','no_later_data_used':True,'definitions':{'isolated':'one spike interval in the day','short':'two or three spike intervals','sustained':'four or more spike intervals, not necessarily consecutive','relative':'exceeds the 99th percentile of the previous 30 complete market days','counterfactual':'Remove D largest interval only for diagnostic threshold calculation; never modify the actual target'},'limitations':'Descriptive exploration; group sizes may be small, groups overlap, no causal inference or new forecasting rule. Oracle gains are unattainable upper bounds.'}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('FREQUENCY');print(pd.DataFrame(frequency).round(3).to_string(index=False));print('PERSISTENCE');print(pd.DataFrame(persistence).round(3).to_string(index=False));print('EPISODES');print(pd.DataFrame(episodes).round(3).to_string(index=False))
print('AFTERDAY METRICS');print(pd.DataFrame(categories).round(3).to_string(index=False));print('THRESHOLD');print(pd.DataFrame(effects).round(3).to_string(index=False));print('GLOBAL IMPACT');print(pd.DataFrame(impacts).round(3).to_string(index=False));print('TIMING');print(pd.DataFrame(timing).round(3).to_string(index=False))
