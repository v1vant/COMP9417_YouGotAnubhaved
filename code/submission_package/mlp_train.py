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


def fit_demand_mlp(features, demand, epochs=5, validation=None, patience=10):
    if features.empty or not features.index.equals(demand.index):
        raise ValueError("Expected nonempty, aligned features and demand")
    if features.shape[1] != 405:
        raise ValueError("Expected the current 405 demand features")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Features must be finite")
    if not np.isfinite(demand.to_numpy()).all() or not demand.gt(0).all():
        raise ValueError("Demand must be finite and positive")
    torch.manual_seed(9417)
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
        generator=torch.Generator().manual_seed(9417),
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


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import load_data, market_days, training_table
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description="Train and export the frozen ten-epoch demand MLP")
    parser.add_argument("csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    selection = json.loads(Path(args.selection).read_text())
    if selection["demand"].get("window_years") or selection["demand"].get("training_end"):
        raise ValueError("Frozen blend requires full released-data expanding-window training")
    with threadpool_limits(limits=2):
        frame = load_data(args.csv)
        features, labels = training_table(frame)
        columns = json.loads((Path(__file__).parent / "demand_feature_columns.json").read_text())
        features = features.loc[:, columns]
        bundle = fit_demand_mlp(features, labels["demand"], epochs=10)
        state = bundle["model"].state_dict()
        arrays = {name.replace(".", "_"): tensor.detach().cpu().numpy()
                  for name, tensor in state.items()}
        arrays.update(mean=bundle["scaler"].mean_, scale=bundle["scaler"].scale_,
                      features=np.array(columns))
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "demand_mlp.npz", **arrays)
        metadata = {"architecture": [405,128,64,1], "epochs": 10, "seed": 9417,
                    "training_rows": len(features), "data_end": str(market_days(frame.index).max()),
                    "torch": torch.__version__, "epoch_seconds": bundle["epoch_seconds"],
                    "inference": "NumPy, no PyTorch import required"}
        # Verify the exported inference path against PyTorch on actual feature rows.
        from model import predict_numpy_mlp
        numpy_prediction = predict_numpy_mlp(arrays, features.iloc[-48:])
        torch_prediction = predict_demand_mlp(bundle, features.iloc[-48:])
        np.testing.assert_allclose(numpy_prediction, torch_prediction, rtol=1e-6, atol=0.002)
        metadata["numpy_torch_max_difference_MW"] = float(np.max(np.abs(numpy_prediction-torch_prediction)))
        (output / "demand_mlp_manifest.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)
