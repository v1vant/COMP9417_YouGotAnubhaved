"""Small CPU demand MLP candidate; independent of the current Forecaster."""
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DemandMLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def fit_demand_mlp(features, demand, epochs=5, validation=None, patience=10, seed=9417):
    if features.empty or not features.index.equals(demand.index):
        raise ValueError("Expected nonempty, aligned features and demand")
    if features.shape[1] != 405:
        raise ValueError("Expected the current 405 demand features")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Features must be finite")
    if not np.isfinite(demand.to_numpy()).all() or not demand.gt(0).all():
        raise ValueError("Demand must be finite and positive")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    scaler = StandardScaler()
    x = scaler.fit_transform(features).astype(np.float32)
    if validation is not None:
        valid_features, valid_demand = validation
        if not features.index.max() < valid_features.index.min():
            raise ValueError("Validation must occur strictly after training")
        if not valid_features.index.equals(valid_demand.index):
            raise ValueError("Validation inputs and targets must align")
        if list(valid_features.columns) != list(features.columns):
            raise ValueError("Validation feature schema must match training")
        if not np.isfinite(valid_features.to_numpy()).all() or not (
                np.isfinite(valid_demand.to_numpy()).all() and valid_demand.gt(0).all()):
            raise ValueError("Validation must be finite with positive demand")
        valid_x = torch.from_numpy(scaler.transform(valid_features).astype(np.float32))
        valid_y = valid_demand.to_numpy()
        valid_anchor = valid_features["demand_anchor"].to_numpy()
    # Predict residuals in units of 1,000 MW for numerical scaling.
    target = (
        (demand.to_numpy() - features["demand_anchor"].to_numpy()) / 1000
    ).astype(np.float32)
    weights = (1 / demand.to_numpy()).astype(np.float32)
    weights /= weights.mean()
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(target),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset, batch_size=1024, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model = DemandMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    timings = []
    history = []
    best_score, best_epoch, best_state, stale = float("inf"), None, None, 0
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        total = 0.0
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (torch.abs(model(batch_x) - batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_x)
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        row = {"epoch": epoch+1, "training_loss": total/len(dataset), "seconds": elapsed}
        if validation is not None:
            model.eval()
            with torch.inference_mode():
                predicted = np.clip(valid_anchor + model(valid_x).numpy()*1000, 0, 20000)
            score = float(np.mean(np.abs(valid_y-predicted)/valid_y)*100)
            row["validation_MAPE"] = score
            if score < best_score:
                best_score, best_epoch, stale = score, epoch+1, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        print(
            f"Epoch {epoch+1}: loss={total/len(dataset):.5f}, "
            f"time={elapsed:.2f}s" + (
                f", validation MAPE={score:.4f}%" if validation is not None else ""), flush=True,
        )
        if validation is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model, "scaler": scaler,
        "features": list(features.columns),
        "epoch_seconds": timings,
        "history": history, "best_epoch": best_epoch, "best_validation_MAPE": (
            best_score if validation is not None else None),
    }


def predict_demand_mlp(bundle, features):
    x = bundle["scaler"].transform(
        features.loc[:, bundle["features"]]).astype(np.float32)
    bundle["model"].eval()
    with torch.inference_mode():
        residual = bundle["model"](torch.from_numpy(x)).numpy() * 1000
    return np.clip(features["demand_anchor"].to_numpy() + residual, 0, 20000)
