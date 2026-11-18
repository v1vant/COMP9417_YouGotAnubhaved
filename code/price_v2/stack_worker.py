"""PyTorch-only subprocess for a strictly chronological auxiliary demand fit."""
import argparse
import numpy as np
import pandas as pd
from mlp_train import fit_demand_mlp, predict_demand_mlp

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('cutoff');p.add_argument('output');a=p.parse_args()
    frame=pd.read_parquet(a.input); y=frame.pop('actual');d=(frame.index-pd.Timedelta(minutes=30)).normalize();train=d<pd.Timestamp(a.cutoff)
    m=fit_demand_mlp(frame.loc[train],y.loc[train],epochs=10)
    np.save(a.output,predict_demand_mlp(m,frame.loc[~train]))
