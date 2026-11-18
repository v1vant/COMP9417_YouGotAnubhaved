"""Actual-price MLP regressor with fixed five/ten-epoch checkpoints."""
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PriceRegressionMLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs,128),nn.ReLU(),
            nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,x):
        return self.network(x).squeeze(-1)


def fit_price_regression_mlp(features, prices, checkpoints=(5,10)):
    if features.empty or not features.index.equals(prices.index):
        raise ValueError("Expected nonempty aligned features and prices")
    if features.shape[1] != 571 or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Expected 571 finite features")
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Actual prices must be finite")
    torch.manual_seed(9417)
    torch.set_num_threads(2)
    scaler = StandardScaler()
    x = scaler.fit_transform(features).astype(np.float32)
    # Scale units for optimisation, retaining negative prices and spikes.
    target = (prices.to_numpy()/100).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(target)),
        batch_size=1024,shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(9417))
    model = PriceRegressionMLP(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    snapshots, history = {}, []
    for epoch in range(1,max(checkpoints)+1):
        started = time.monotonic()
        model.train()
        total = 0.
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.abs(model(bx)-by).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()*len(bx)
        row = {"epoch":epoch,"training_L1_dollars_per_MWh":total/len(target)*100,
               "seconds":time.monotonic()-started}
        history.append(row)
        if epoch in checkpoints:
            snapshots[epoch] = {k:v.detach().clone() for k,v in model.state_dict().items()}
            print(row,flush=True)
    return {"scaler":scaler,"features":list(features.columns),"states":snapshots,"history":history}


def predict_price_regression_mlp(bundle,features,epoch):
    model = PriceRegressionMLP(len(bundle["features"]))
    model.load_state_dict(bundle["states"][epoch])
    model.eval()
    x = bundle["scaler"].transform(features.loc[:,bundle["features"]]).astype(np.float32)
    with torch.inference_mode():
        return model(torch.from_numpy(x)).numpy()*100
