"""Pure test (not in the repo): the package's numpy MLP (train.fit_neural_net) with two optional extras.

  late:   extra columns joined to the LAST hidden layer, so they feed the output neuron directly
          ("demand forecasts as neurons" rather than as first-layer inputs);
  offset: a fixed per-row logit g (LightGBM's, out-of-sample) added to the output neuron with a
          learnable weight a (initialised to 1, or frozen at 1 when offset_learn is False):
              logit = W_out . [h_last, late] + b + a * g
          so the network starts at LightGBM's prediction and learns a correction to it.

With neither extra, fit() performs exactly the same computation as train.fit_neural_net (same random
draws, same updates), which test_reproduces_package() checks."""
import numpy as np
import pandas as pd
from model import market_days, neural_net_columns, neural_net_log_mask, neural_net_transform, neural_net_inputs


def loss_gradients(layers, x, y, pos_weight, masks=None, late=None, g=None, a=None):
    outputs, h = [x], x
    for index, (weight, bias) in enumerate(layers[:-1]):
        h = np.maximum(h @ weight.T + bias, 0)
        if masks is not None:
            h = h * masks[index]
        outputs.append(h)
    weight, bias = layers[-1]
    top = h if late is None else np.concatenate([h, late], axis=1)
    logit = (top @ weight.T + bias)[:, 0]
    if g is not None:
        logit = logit + a * g
    row_weight = 1 + (pos_weight - 1) * y
    loss = np.mean(pos_weight * y * np.logaddexp(0, -logit) + (1 - y) * np.logaddexp(0, logit))
    probability = 1 / (1 + np.exp(-np.clip(logit, -60, 60)))
    delta = (row_weight * (probability - y) / len(y))[:, None]
    grad_a = None if g is None else np.float32((delta[:, 0] * g).sum())
    gradients = [(delta.T @ top, delta.sum(axis=0))]
    hidden = h.shape[1]
    delta = delta @ weight[:, :hidden]
    for index in reversed(range(len(layers) - 1)):
        if masks is not None:
            delta = delta * masks[index]
        delta = delta * (outputs[index + 1] > 0)
        gradients.append((delta.T @ outputs[index], delta.sum(axis=0)))
        if index > 0:
            delta = delta @ layers[index][0]
    return loss, gradients[::-1], grad_a


def _standardise(values):
    mean, std = values.mean(axis=0), values.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def fit(config, features, labels, late=None, offset=None, offset_learn=True, a_lr_mult=1.0, z_clip=None):
    log = config.get("log", "price")
    columns = neural_net_columns(features, log)
    params = {"columns": columns, "log_mask": neural_net_log_mask(columns, log), "clip_low": None, "clip_high": None}
    x = neural_net_transform(params, features)
    assert not config.get("winsorise"), "not supported here"
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-8] = 1.0
    inputs = ((x - mean) / std).astype(np.float32)
    if z_clip:              # cap standardised inputs: unprecedented values (regime shifts) cannot swing the network
        np.clip(inputs, -z_clip, z_clip, out=inputs)
    params.update(mean=mean, std=std, z_clip=z_clip)
    if late is not None:
        params["late_mean"], params["late_std"] = _standardise(np.asarray(late, np.float64))
        late = ((late - params["late_mean"]) / params["late_std"]).astype(np.float32)
    g = None if offset is None else np.asarray(offset, np.float32)
    y = labels.to_numpy().astype(np.float32) if hasattr(labels, "to_numpy") else np.asarray(labels, np.float32)
    rows = len(y)
    day_codes = pd.factorize(market_days(features.index))[0]
    day_rows = np.split(np.argsort(day_codes, kind="stable"), np.cumsum(np.bincount(day_codes))[:-1])
    batch = min(config["batch_size"], rows)
    pos_weight = np.float32((y == 0).sum() / (y == 1).sum())
    keep = np.float32(1 - config["dropout"])
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    extra = 0 if late is None else late.shape[1]
    sizes = [len(columns)] + list(config["hidden"]) + [1]
    members, scales = [], []
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)
        layers = []
        for i, (units_in, units_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            if i == len(sizes) - 2:
                units_in += extra                                  # output neuron also sees the late columns
            bound = 1 / np.sqrt(units_in)
            layers.append([rng.uniform(-bound, bound, (units_out, units_in)).astype(np.float32),
                           rng.uniform(-bound, bound, units_out).astype(np.float32)])
        first = [[np.zeros_like(p) for p in layer] for layer in layers]
        second = [[np.zeros_like(p) for p in layer] for layer in layers]
        a, a_m, a_v = np.float32(1.0), 0.0, 0.0
        sample = (np.concatenate([day_rows[d] for d in rng.integers(0, len(day_rows), len(day_rows))])
                  if config.get("bootstrap") else np.arange(rows))
        steps_per_epoch = len(sample) // batch
        total_steps = config["epochs"] * steps_per_epoch
        step = 0
        for _ in range(config["epochs"]):
            order = sample[rng.permutation(len(sample))]
            for start in range(0, steps_per_epoch * batch, batch):
                chosen = order[start:start + batch]
                masks = [(rng.random((batch, units), dtype=np.float32) < keep) / keep for units in config["hidden"]]
                _, gradients, grad_a = loss_gradients(
                    layers, inputs[chosen], y[chosen], pos_weight, masks,
                    None if late is None else late[chosen], None if g is None else g[chosen], a)
                rate = config["learning_rate"] * 0.5 * (1 + np.cos(np.pi * step / total_steps))
                step += 1
                for layer, grads, means, squares in zip(layers, gradients, first, second):
                    for k, gradient in enumerate(grads):
                        means[k] = beta1 * means[k] + (1 - beta1) * gradient
                        squares[k] = beta2 * squares[k] + (1 - beta2) * gradient * gradient
                        if k == 0:
                            layer[k] *= 1 - rate * config["weight_decay"]
                        layer[k] -= (rate * (means[k] / (1 - beta1 ** step))
                                     / (np.sqrt(squares[k] / (1 - beta2 ** step)) + epsilon)).astype(np.float32)
                if g is not None and offset_learn:                 # Adam on the scalar a, no weight decay
                    a_m = beta1 * a_m + (1 - beta1) * float(grad_a)
                    a_v = beta2 * a_v + (1 - beta2) * float(grad_a) ** 2
                    a = np.float32(a - a_lr_mult * rate * (a_m / (1 - beta1 ** step)) / (np.sqrt(a_v / (1 - beta2 ** step)) + epsilon))
        members.append([(w, b) for w, b in layers])
        scales.append(float(a))
    params.update(members=members, offset_scales=scales, log=log, winsorise=None)
    return params


def predict(params, features, late=None, offset=None):
    z = neural_net_inputs(params, features)
    if params.get("z_clip"):
        z = np.clip(z, -params["z_clip"], params["z_clip"])
    if late is not None:
        late = (np.asarray(late, np.float64) - params["late_mean"]) / params["late_std"]
    total = np.zeros(len(z))
    for layers, a in zip(params["members"], params["offset_scales"]):
        h = z
        for weight, bias in layers[:-1]:
            h = np.maximum(h @ weight.T + bias, 0.0)
        weight, bias = layers[-1]
        top = h if late is None else np.concatenate([h, late], axis=1)
        logit = (top @ weight.T + bias)[:, 0]
        if offset is not None:
            logit = logit + a * np.asarray(offset, np.float64)
        total += 1 / (1 + np.exp(-np.clip(logit, -60.0, 60.0)))
    return total / len(params["members"])
