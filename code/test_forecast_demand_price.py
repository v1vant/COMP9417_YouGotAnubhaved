"""Isolated 2014 price screen using chronological frozen demand-ensemble forecasts."""
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/forecast_demand_price_2014'


def mlp_worker(source, destination):
    # No LightGBM artifacts or imports in this Torch process.
    from demand_mlp import fit_demand_mlp, predict_demand_mlp
    train_x, train_y, test_x = pd.read_pickle(source)
    bundle = fit_demand_mlp(train_x, train_y, epochs=10)
    pd.Series(predict_demand_mlp(bundle, test_x), index=test_x.index).to_csv(
        destination, index_label='timestamp', header=['mlp'])


def added_features(x, prediction):
    return pd.DataFrame({
        'forecast_demand': prediction,
        'forecast_demand_minus_yesterday_slot': prediction-x.nsw_demand_lag_1,
        'forecast_demand_minus_lastweek_slot': prediction-x.nsw_demand_lag_7,
        'forecast_demand_minus_yesterday_mean': prediction-x.nsw_demand_previous_mean,
        'forecast_demand_minus_yesterday_max': prediction-x.nsw_demand_previous_max,
    }, index=x.index)


def main():
    from run_experiments import load_data, market_days, training_table, fit_model
    from forecast_package.model import predict_task
    from test_price_recency_weights import metrics
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'completion.json').exists():
        print((OUT/'metrics.csv').read_text(), flush=True)
        return
    selection = json.loads((ROOT/'results/final_config.json').read_text())
    columns = json.loads((ROOT/'code/demand_feature_columns.json').read_text())
    frame = load_data(ROOT/'data/nem_weather_halfhourly_2010_2026.csv').loc[:'2015-01-01 00:00:00']
    x, y = training_table(frame)
    assert x.shape[1] == 571 and len(columns) == 405
    days = market_days(x.index)
    forecasts = []
    audit = []
    for year in range(2011, 2015):
        train, test = days.year < year, days.year == year
        assert days[train].max() < days[test].min()
        destination = OUT/f'demand_forecasts_{year}.csv'
        if destination.exists():
            fold = pd.read_csv(destination, index_col='timestamp', parse_dates=True)
            assert fold.index.equals(x.index[test])
        else:
            print(f'Demand ensemble fold {year}: training through {year-1}', flush=True)
            source = OUT/f'mlp_input_{year}.pkl'
            mlp_path = OUT/f'mlp_forecast_{year}.csv'
            if not mlp_path.exists():
                pd.to_pickle((x.loc[train, columns], y.loc[train, 'demand'], x.loc[test, columns]), source)
                subprocess.run([sys.executable, '-u', __file__, '--mlp-worker', str(source), str(mlp_path)], check=True)
                source.unlink()
            mlp = pd.read_csv(mlp_path, index_col='timestamp', parse_dates=True).mlp
            assert mlp.index.equals(x.index[test])
            model = fit_model('demand', selection['demand']['config'], x.loc[train], y.loc[train])
            assert model['features'] == columns
            lgbm = predict_task(model, x.loc[test])
            fold = pd.DataFrame({'lgbm': lgbm, 'mlp': mlp}, index=x.index[test])
            fold['forecast'] = .5*(fold.lgbm+fold.mlp)
            assert np.isfinite(fold.to_numpy()).all()
            fold.to_csv(destination, index_label='timestamp')
            joblib.dump(model, OUT/f'demand_lgbm_{year}.joblib')
        forecasts.append(fold)
        audit.append({'prediction_year':year, 'training_end':str(days[train].max().date()),
                      'training_rows':int(train.sum()), 'MLP_epochs':10, 'LGBM_weight':.5})
        pd.DataFrame(audit).to_csv(OUT/'chronology_audit.csv', index=False)
        print(f'Saved demand ensemble forecasts for {year}', flush=True)
    prediction = pd.concat(forecasts).forecast
    extra = added_features(x.loc[prediction.index], prediction)
    assert np.isfinite(extra.to_numpy()).all()
    train = market_days(extra.index).year < 2014
    test = ~train
    train_index, test_index = extra.index[train], extra.index[test]
    config = selection['price']['config']
    rows = []
    output = pd.DataFrame({'actual_up': y.loc[test_index,'price']}, index=test_index)
    cached = pd.read_csv(ROOT/'results/price_recency_2014/predictions.csv', index_col='timestamp', parse_dates=True)
    assert cached.index.equals(test_index)
    np.testing.assert_array_equal(cached.actual_up, output.actual_up)
    output['current_probability'] = cached.current_probability
    rows.append(dict(model='current_2010_2013', **metrics(output.actual_up, cached.current_probability > config['threshold'])))
    for name, features in [('matched_2011_2013',x.loc[extra.index]),
                           ('forecast_demand_augmented',x.loc[extra.index].join(extra))]:
        print(f'Fitting price classifier: {name}', flush=True)
        model = fit_model('price', config, features.loc[train_index], y.loc[train_index])
        probability = model['estimator'].predict_proba(features.loc[test_index,model['features']])[:,1]
        output[name+'_probability'] = probability
        rows.append(dict(model=name, **metrics(output.actual_up,probability > config['threshold'])))
        joblib.dump(model, OUT/f'{name}.joblib')
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv', index=False)
        output.to_csv(OUT/'predictions.csv', index_label='timestamp')
    (OUT/'completion.json').write_text(json.dumps({'seconds':time.monotonic()-started,
        'features_added':list(extra), 'validation_year':2014,'price_config':config,
        'classifier_training_years':[2011,2012,2013], 'holdout_evaluated':False}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        if len(sys.argv)>1 and sys.argv[1]=='--mlp-worker':
            mlp_worker(sys.argv[2], sys.argv[3])
        else:
            main()
