"""Small CPU price classifier, separate from the current LightGBM package."""
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class PriceMLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def fit_price_mlp(features, labels, epochs=5, validation=None, patience=10):
    if features.empty or not features.index.equals(labels.index):
        raise ValueError("Expected nonempty aligned features and labels")
    if features.shape[1] != 571 or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Expected 571 finite features")
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Training requires both binary classes")
    torch.manual_seed(9417)
    torch.set_num_threads(2)
    scaler = StandardScaler()
    x = scaler.fit_transform(features).astype(np.float32)
    if validation is not None:
        valid_features, valid_labels = validation
        if not features.index.max() < valid_features.index.min():
            raise ValueError("Validation must occur strictly after training")
        if not valid_features.index.equals(valid_labels.index):
            raise ValueError("Validation inputs and labels must align")
        if list(valid_features.columns) != list(features.columns):
            raise ValueError("Validation schema must match training")
        if not np.isfinite(valid_features.to_numpy()).all() or set(valid_labels.unique()) != {0,1}:
            raise ValueError("Expected finite validation features and both binary classes")
        valid_x = torch.from_numpy(scaler.transform(valid_features).astype(np.float32))
        valid_y = valid_labels.to_numpy()
    y = labels.to_numpy(dtype=np.float32)
    counts = np.bincount(y.astype(int), minlength=2)
    class_weights = len(y)/(2*counts)
    weights = class_weights[y.astype(int)].astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y),
        torch.from_numpy(weights)), batch_size=1024, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(9417))
    model = PriceMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    history = []
    best_score, best_epoch, best_state, stale = -float("inf"), None, None, 0
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        total = 0.0
        for bx, by, bw in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_fn(model(bx), by)*bw).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()*len(bx)
        row = {"epoch": epoch+1, "weighted_BCE": total/len(y),
               "seconds": time.monotonic()-started}
        if validation is not None:
            model.eval()
            with torch.inference_mode():
                probability = torch.sigmoid(model(valid_x)).numpy()
            score = float(balanced_accuracy_score(valid_y, probability>0.5)*100)
            row["validation_balanced_accuracy"] = score
            if score > best_score:
                best_score, best_epoch, stale = score, epoch+1, 0
                best_state = {key:value.detach().clone() for key,value in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        print(row, flush=True)
        if validation is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "scaler": scaler, "features": list(features.columns),
            "class_weights": class_weights.tolist(), "history": history,
            "best_epoch": best_epoch, "best_validation_balanced_accuracy": (
                best_score if validation is not None else None)}


def predict_price_probability(bundle, features):
    x = bundle["scaler"].transform(features.loc[:, bundle["features"]]).astype(np.float32)
    bundle["model"].eval()
    with torch.inference_mode():
        return torch.sigmoid(bundle["model"](torch.from_numpy(x))).numpy()
