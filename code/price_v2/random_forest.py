"""Random Forest candidates and estimators for the two forecasting tasks.

Feature construction, residual targets, fitting weights, and chronological
evaluation are shared with the other algorithms through train.py.

Select this module with --families random_forest in train.py or
run_experiments.py. It fits separate forests for demand and price direction.
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def candidates(quick=False):
    """Starting settings to compare on 2014 validation data."""
    configs = [{"family": "random_forest", "trees": 20 if quick else 200,
                "max_depth": 12, "min_samples_leaf": 20, "max_features": 0.7}]
    if not quick:
        configs.append({"family": "random_forest", "trees": 200,
                        "max_depth": 20, "min_samples_leaf": 40,
                        "max_features": 0.7})
    return configs


def make_estimator(task, config, seed=9417):
    """Build an unfitted forest; demand predicts residuals, price predicts labels."""
    if task not in ("demand", "price"):
        raise ValueError(f"Unknown forecasting task: {task}")
    cls = RandomForestRegressor if task == "demand" else RandomForestClassifier
    return cls(
        n_estimators=config["trees"], max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"], bootstrap=True,
        n_jobs=2, random_state=seed,
        **({"class_weight": "balanced"} if task == "price" else {}),
    )
