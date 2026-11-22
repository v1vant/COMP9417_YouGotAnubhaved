import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from common import (read_json,build_windows,verify_sources,cache_location,fit_snapshot,
                    predict_week,score_week,validate_cache,frozen_model,harness)
from uncertainty import bootstrap_indices,summarize


class ScheduleTests(unittest.TestCase):
    def test_seasonal_and_recent_windows_match_live_age_and_do_not_overlap(self):
        table=build_windows()
        self.assertEqual(len(table),21)
        self.assertEqual(int((table.group=='seasonal').sum()),11)
        self.assertEqual(int((table.group=='recent').sum()),10)
        seasonal=table.loc[table.group=='seasonal']
        self.assertEqual(list(pd.to_datetime(seasonal.week_start).dt.year),list(range(2015,2026)))
        self.assertTrue(seasonal.week_start.str.endswith('-12-07').all())
        self.assertTrue(seasonal.week_end.str.endswith('-12-13').all())
        recent=table.loc[table.group=='recent']
        self.assertEqual(recent.iloc[0].week_start,'2025-02-06')
        self.assertEqual(recent.iloc[-1].week_end,'2025-11-12')
        for row in table.itertuples():
            self.assertEqual((pd.Timestamp(row.week_start)-pd.Timestamp(row.training_cutoff)).days,37)
            self.assertEqual((pd.Timestamp(row.week_end)-pd.Timestamp(row.week_start)).days,6)

    def test_frozen_sources_and_current_recipe(self):
        verify_sources();recipe=read_json(HERE/'recipe.json')
        self.assertEqual(recipe['demand']['mlp_seeds'],list(range(9417,9422)))
        self.assertEqual(recipe['price']['config']['seeds'],list(range(9417,9422)))
        self.assertEqual(recipe['price']['config']['threshold'],.5)

    def test_smoke_cache_cannot_alias_full_evaluation(self):
        window=build_windows().iloc[0].to_dict()
        self.assertNotEqual(cache_location(window,'same',quick=True),cache_location(window,'same',quick=False))


class UncertaintyTests(unittest.TestCase):
    def weekly(self,n):
        rows=[]
        for week in range(n):
            for model in ['current','baseline']:
                current=model=='current'
                rows.append(dict(group='recent',model=model,week_start=str((pd.Timestamp('2025-01-01')+pd.Timedelta(days=30*week)).date()),
                    demand_MAPE=8+week if current else 10+week,demand_MAE=80+week if current else 100+week,
                    price_accuracy=75+week if current else 70+week,price_balanced_accuracy=76+week if current else 70+week))
        return pd.DataFrame(rows)

    def test_adjacent_pair_bootstrap_preserves_pairs_and_is_reproducible(self):
        a=bootstrap_indices(10,100,9417,2);b=bootstrap_indices(10,100,9417,2)
        self.assertEqual(a.shape,(100,10));np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,1::2],(a[:,::2]+1)%10)

    def test_paired_improvement_direction_and_percentiles(self):
        settings=read_json(HERE/'plan.json')['bootstrap']
        summary,paired,sensitivity=summarize(self.weekly(10),settings)
        expected={'demand_MAPE':2.,'demand_MAE':20.,'price_accuracy':5.,'price_balanced_accuracy':6.}
        for row in paired.itertuples():
            self.assertEqual(row.mean_improvement,expected[row.metric])
            self.assertAlmostEqual(row.paired_CI95_low,expected[row.metric])
            self.assertAlmostEqual(row.paired_CI95_high,expected[row.metric])
        self.assertEqual(len(summary),8);self.assertEqual(len(sensitivity),12)
        current=summary.loc[(summary.model=='current')&(summary.metric=='demand_MAPE')].iloc[0]
        self.assertEqual(current['mean'],12.5)
        self.assertAlmostEqual(current.empirical_week_p10,8.9)
        self.assertAlmostEqual(current.empirical_week_p90,16.1)

    def test_single_pilot_week_does_not_emit_a_confidence_interval(self):
        summary,paired,_=summarize(self.weekly(1),read_json(HERE/'plan.json')['bootstrap'])
        self.assertTrue(summary.mean_CI95_low.isna().all())
        self.assertTrue(summary.empirical_week_p10.isna().all())
        self.assertTrue(paired.paired_CI95_high.isna().all())

    def test_reporting_graphs_are_written_separately(self):
        import run as live_run
        weekly=self.weekly(10)
        summary,_,_=summarize(weekly,read_json(HERE/'plan.json')['bootstrap'])
        with tempfile.TemporaryDirectory(prefix='live_week_plot_test_') as directory:
            with patch.object(live_run,'HERE',Path(directory)):
                live_run.plot(weekly,summary)
            for name in ['recent_weekly_performance','summary_intervals']:
                for suffix in ['png','pdf']:
                    path=Path(directory)/'figures'/f'{name}.{suffix}'
                    self.assertTrue(path.exists());self.assertGreater(path.stat().st_size,0)


class SyntheticSmokeTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec('torch') is None,'PyTorch needed for clean training smoke')
    def test_actual_training_daily_inference_cache_and_baseline(self):
        # No real historical fits: a synthetic prefix plus a 37-day training gap.
        model=frozen_model();da=harness()
        index=pd.date_range('2015-09-01 00:30',periods=160*48,freq='30min')
        slot=np.arange(len(index))%48;day=np.arange(len(index))//48
        frame=pd.DataFrame(index=index)
        for number,column in enumerate(model.MARKET):
            frame[column]=(7000+400*np.sin(slot*2*np.pi/48)+day*3+number if column.endswith('demand')
                           else 40+30*np.cos(slot*2*np.pi/48)+day)
        for number,column in enumerate(model.WEATHER):frame[column]=20+number+5*np.sin(slot*2*np.pi/48)
        cutoff=pd.Timestamp('2015-11-30');start=cutoff+pd.Timedelta(days=37)
        window=dict(window='synthetic_smoke',group='smoke',training_cutoff=str(cutoff.date()),
            week_start=str(start.date()),week_end=str((start+pd.Timedelta(days=6)).date()))
        artifacts=ROOT/'code/forecast_package/artifacts'
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.iterdir() if p.is_file()}
        with tempfile.TemporaryDirectory(prefix='live_week_smoke_') as directory:
            path=fit_snapshot(frame,window,'synthetic-test-only',Path(directory)/'cache',Path(directory)/'logs',quick=True)
            metadata=read_json(path/'complete.json')
            self.assertEqual(metadata['training_cutoff'],'2015-11-30')
            self.assertEqual(metadata['training_raw_rows'],int((da.market_day(frame.index)<=cutoff).sum()))
            manifest=read_json(path/'package/artifacts/manifest.json')
            self.assertTrue(manifest['quick']);self.assertEqual(len(manifest['demand_mlp_members']),5)
            predictions=predict_week(frame,window,path,'synthetic-test-only',quick=True)
            self.assertEqual(len(predictions),336)
            self.assertEqual(set(predictions.actual_nsw_price_up.unique()),{0,1})
            rows=score_week(predictions);self.assertEqual({row['model'] for row in rows},{'current','baseline'})
            for row in rows:
                self.assertTrue(np.isfinite(row['demand_MAPE']))
                self.assertTrue(0<=row['price_balanced_accuracy']<=100)
            # Full mode rejects even a valid smoke artifact.
            self.assertFalse(validate_cache(path,'synthetic-test-only',quick=False))
            # A changed artifact is rejected, so resume cannot silently use it.
            target=path/'package/artifacts/manifest.json';target.write_text(target.read_text()+' ')
            self.assertFalse(validate_cache(path,'synthetic-test-only',quick=True))
        after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.iterdir() if p.is_file()}
        self.assertEqual(before,after)


if __name__=='__main__':unittest.main()
