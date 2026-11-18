"""One out-of-sample fold of Anubhav's ten-epoch demand MLP (code/demand_mlp.fit_demand_mlp, unchanged except that the
405-column check is dropped so extra inputs can be added). NaNs (only the solar-sensitivity inputs in the first weeks of
2010) are filled with the training-fold means. Usage: mlp_fold.py IN.pkl OUT.npy"""
import inspect, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # code/
import demand_mlp
from threadpoolctl import threadpool_limits

src = inspect.getsource(demand_mlp.fit_demand_mlp)
check = '    if features.shape[1] != 405:\n        raise ValueError("Expected the current 405 demand features")\n'
assert check in src
ns = dict(vars(demand_mlp))
exec(src.replace(check, ""), ns)
fit = ns["fit_demand_mlp"]

xtr, ytr, xte = pd.read_pickle(sys.argv[1])
means = xtr.mean()
xtr, xte = xtr.fillna(means), xte.fillna(means)
with threadpool_limits(limits=2):
    bundle = fit(xtr, ytr, epochs=10)
    np.save(sys.argv[2], demand_mlp.predict_demand_mlp(bundle, xte).astype(np.float64))
