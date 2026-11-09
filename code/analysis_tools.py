"""Statistical analysis used by the chronological experiments."""
from collections import deque

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

SEED = 9417


def metric_values(demand, price, predicted_demand, predicted_price):
    demand = np.asarray(demand, dtype=float)
    price = np.asarray(price, dtype=int)
    predicted_demand = np.asarray(predicted_demand, dtype=float)
    predicted_price = np.asarray(predicted_price, dtype=int)
    recalls = [np.mean(predicted_price[price == label] == label)
               for label in np.unique(price)]
    return {
        "MAPE": float(np.mean(np.abs(demand - predicted_demand) / demand) * 100),
        "MAE": float(np.mean(np.abs(demand - predicted_demand))),
        "accuracy": float(np.mean(price == predicted_price) * 100),
        "balanced_accuracy": float(np.mean(recalls) * 100),
    }


def confidence_intervals(errors, correct, bootstrap=1000, seed=SEED):
    """Inputs have shape (days, 48); resampling day means is exactly equivalent."""
    errors = np.asarray(errors, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != 48 or errors.shape != correct.shape:
        raise ValueError("Expected aligned arrays with one complete market day per row")
    if bootstrap < 1000:
        raise ValueError("The specification requires at least 1000 bootstrap draws")
    n = errors.size
    mape = errors.mean()
    p = correct.mean()
    normal_mape = 1.96 * errors.std(ddof=1) / np.sqrt(n)
    normal_accuracy = 1.96 * np.sqrt(p * (1 - p) / n)
    hoeffding = np.sqrt(np.log(40) / (2 * n))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(errors), size=(bootstrap, len(errors)))
    demand_boot = errors.mean(axis=1)[draws].mean(axis=1)
    accuracy_boot = correct.mean(axis=1)[draws].mean(axis=1) * 100
    demand_low, demand_high = np.percentile(demand_boot, [2.5, 97.5])
    accuracy_low, accuracy_high = np.percentile(accuracy_boot, [2.5, 97.5])
    return {
        "MAPE_iid_low": float(mape - normal_mape),
        "MAPE_iid_high": float(mape + normal_mape),
        "accuracy_iid_low": float(max(0, p - normal_accuracy) * 100),
        "accuracy_iid_high": float(min(1, p + normal_accuracy) * 100),
        "accuracy_hoeffding_low": float(max(0, p - hoeffding) * 100),
        "accuracy_hoeffding_high": float(min(1, p + hoeffding) * 100),
        "MAPE_day_boot_low": float(demand_low),
        "MAPE_day_boot_high": float(demand_high),
        "accuracy_day_boot_low": float(accuracy_low),
        "accuracy_day_boot_high": float(accuracy_high),
    }


class OnlineDetector:
    """A 28-day rolling mean; threshold calibrated on 2014 before deployment."""
    def __init__(self, reference, window=28, quantile=0.99, cooldown=28):
        reference = np.asarray(reference, dtype=float)
        rolling = pd.Series(reference).rolling(window).mean().dropna()
        if rolling.empty:
            raise ValueError("Not enough reference days for detector calibration")
        self.threshold = float(rolling.quantile(quantile))
        self.window = window
        self.values = deque(reference[-(window - 1):], maxlen=window)
        self.cooldown = cooldown
        self.last_signal = None
        self.statistic = None

    def observe(self, day, value):
        self.values.append(float(value))
        self.statistic = float(np.mean(self.values))
        active = len(self.values) == self.window and self.statistic > self.threshold
        eligible = self.last_signal is None or (day - self.last_signal).days >= self.cooldown
        signal = active and eligible
        if signal:
            self.last_signal = day
        return signal


def day_type_analysis(frame, normalize=True):
    days = (frame.index - pd.Timedelta(minutes=30)).normalize()
    unique_days = days.unique()
    profiles = frame["nsw_demand"].to_numpy().reshape(-1, 48)
    inputs = profiles / profiles.mean(axis=1, keepdims=True) if normalize else profiles
    training = unique_days.year <= 2013
    pca = PCA(n_components=0.95, svd_solver="full")
    reference = pca.fit_transform(inputs[training])
    scores, models = {}, {}
    for k in range(2, 7):
        clustering = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        labels = clustering.fit_predict(reference)
        scores[k] = float(silhouette_score(reference, labels,
                                          sample_size=min(1000, len(reference)),
                                          random_state=SEED))
        models[k] = clustering
    k = max(scores, key=scores.get)
    clustering = models[k]
    embedding = pca.transform(inputs)
    labels = clustering.predict(embedding)
    distance = np.min(clustering.transform(embedding), axis=1)
    reconstruction = np.linalg.norm(inputs - pca.inverse_transform(embedding), axis=1)
    # Include orthogonal changes that PCA would otherwise discard.
    drift_score = np.sqrt(distance ** 2 + reconstruction ** 2)
    typical = pca.inverse_transform(clustering.cluster_centers_)
    table = pd.DataFrame({
        "day": unique_days, "year": unique_days.year, "cluster": labels,
        "mean_demand": profiles.mean(axis=1), "distance": distance,
        "reconstruction_error": reconstruction, "input_drift_score": drift_score,
    }).set_index("day")
    shares = pd.crosstab(table["year"], table["cluster"], normalize="index")
    return table, shares, typical, {
        "pca": pca, "clustering": clustering, "normalize": normalize,
        "silhouette_scores": scores, "selected_clusters": k,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "reference": "2010-2013",
    }


def period_metrics(days, truth, predictions, bootstrap=1000):
    rows = []
    for frequency in ("M", "Y"):
        periods = days.to_period(frequency)
        for period in periods.unique():
            mask = periods == period
            y = truth["demand"][mask]
            c = truth["price"][mask]
            yd = predictions["demand"][mask]
            cp = predictions["price"][mask]
            metrics = metric_values(y.ravel(), c.ravel(), yd.ravel(), cp.ravel())
            row = {"frequency": frequency, "period": str(period), "days": int(mask.sum()),
                   **metrics}
            if frequency == "M":
                row.update(confidence_intervals(100 * np.abs(y - yd) / y, c == cp,
                                                bootstrap=bootstrap))
            rows.append(row)
    return pd.DataFrame(rows)


def save_figures(output, metrics, types, shares, typical, predictions, days, truth):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = os.path.join(output, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, profile in enumerate(typical):
        ax.plot(np.arange(48) / 2, profile, label=f"Cluster {k}")
    ax.set(xlabel="Interval start (AEST hours)",
           ylabel="Demand / daily mean (unitless)", title="Fixed training-period day types")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "day_type_profiles.png"), dpi=160)
    plt.close(fig)
    ax = shares.plot.area(figsize=(10, 5))
    ax.set(xlabel="Year", ylabel="Share of days (fraction)", title="Day-type prevalence")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(figure_dir, "day_type_shares.png"), dpi=160)
    plt.close(ax.figure)
    monthly = metrics[metrics["frequency"] == "M"]
    for name in monthly["model"].unique():
        subset = monthly[monthly["model"] == name]
        dates = pd.to_datetime(subset["period"] + "-01")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for ax, metric, ylabel in zip(axes, ("MAPE", "accuracy"),
                                     ("Demand MAPE (%)", "Price accuracy (%)")):
            ax.plot(dates, subset[metric], color="black", linewidth=1, label=metric)
            for interval, color in (("iid", "tab:orange"), ("day_boot", "tab:blue")):
                ax.fill_between(dates, subset[f"{metric}_{interval}_low"],
                                subset[f"{metric}_{interval}_high"], alpha=0.25,
                                color=color, label=f"95% {interval}")
            ax.set_ylabel(ylabel)
            ax.legend(loc="best")
        axes[-1].set_xlabel("Month")
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(os.path.join(figure_dir, f"monthly_{name}.png"), dpi=150)
        plt.close(fig)
    selected = days >= days.max() - pd.Timedelta(days=6)
    index = pd.date_range(days[selected][0] + pd.Timedelta(minutes=30),
                          periods=int(selected.sum()) * 48, freq="30min")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(index, truth["demand"][selected].ravel(), label="Observed", color="black")
    axes[1].plot(index, truth["price"][selected].ravel(), label="Observed", color="black")
    for name, values in predictions.items():
        if name.startswith("adapted_"):
            axes[0].plot(index, values["demand"][selected].ravel(), alpha=0.6, label=name)
            axes[1].plot(index, values["price"][selected].ravel(), alpha=0.4, label=name)
    axes[0].set_ylabel("NSW demand (MW)")
    axes[1].set_ylabel("Price direction (0 / 1)")
    axes[1].set_xlabel("Interval end (AEST)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "last_week_predictions.png"), dpi=160)
    plt.close(fig)
