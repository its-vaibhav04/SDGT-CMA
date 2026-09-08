"""Gradient-boosted tree baseline.

This one is not optional. A July 2026 study on this exact dataset (arXiv
2607.07279, same 420,768 rows, same 12 stations) reports XGBoost as the best
model at the 6-hour horizon and an elastic net at 24 hours -- no deep model was
needed to win anything. A thesis that omits a gradient-boosted comparator has a
hole a viva will find immediately.

One model per horizon, which is the standard direct multi-horizon setup for
trees. Features are in physical units for the same reason as the linear
baselines: trees are scale-invariant but not representation-invariant, and
handing them the log-scaled column against a raw target adds nothing.
"""

from __future__ import annotations

import numpy as np

from src.baselines.linear import LINEAR_FEATURES, _design, _targets, physical_features
from src.data.contract import ProcessedDataset
from src.metrics import Predictions

DEFAULT_PARAMS = {
    "objective": "regression_l1",   # matches the MAE the project reports
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 100,
    "verbose": -1,
}


def lightgbm_direct(
    dataset: ProcessedDataset,
    origins: dict[str, np.ndarray],
    *,
    lookback: int = 48,
    horizon: int = 24,
    n_estimators: int = 400,
    early_stopping: int = 30,
    seed: int = 0,
    params: dict | None = None,
) -> Predictions:
    """One LightGBM regressor per horizon, early-stopped on the validation split."""
    import lightgbm as lgb

    settings = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
    values = physical_features(dataset, LINEAR_FEATURES)
    n_stations = dataset.n_stations

    Xtr = _design(values, origins["train"], lookback, n_stations)
    Xva = _design(values, origins["val"], lookback, n_stations)
    Xte = _design(values, origins["test"], lookback, n_stations)

    _, _, Ytr, Wtr = _targets(dataset, origins["train"], horizon)
    _, _, Yva, Wva = _targets(dataset, origins["val"], horizon)
    truth, mask, _, _ = _targets(dataset, origins["test"], horizon)

    predictions = np.zeros((len(Xte), horizon), dtype=np.float64)
    for h in range(horizon):
        train_rows, val_rows = Wtr[:, h], Wva[:, h]
        train_set = lgb.Dataset(Xtr[train_rows], label=Ytr[train_rows, h])
        val_set = lgb.Dataset(Xva[val_rows], label=Yva[val_rows, h], reference=train_set)

        booster = lgb.train(
            settings,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        predictions[:, h] = booster.predict(Xte, num_iteration=booster.best_iteration)

    pred = predictions.reshape(len(origins["test"]), n_stations, horizon)
    return Predictions(
        origins=origins["test"],
        pred=pred.astype(np.float32),
        truth=truth.astype(np.float32),
        mask=mask,
        model="lightgbm",
        seed=seed,
    )
