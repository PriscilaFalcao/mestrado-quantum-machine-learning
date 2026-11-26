from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def influence_selection(
    X: np.ndarray,
    y: np.ndarray,
    poison_fraction: float,
    random_seed: int,
) -> np.ndarray:
    """
    Influence-inspired heuristic:
    prioritize points with high leverage and high residual.
    """
    poison_count = max(1, int(np.floor(len(X) * poison_fraction)))

    ridge = Ridge(alpha=1.0, random_state=random_seed)
    ridge.fit(X, y)
    pred = ridge.predict(X)

    residual = np.abs(pred - y)
    leverage = np.linalg.norm(X, axis=1)
    scores = residual * leverage

    top_idx = np.argsort(scores)[-poison_count:]
    return np.sort(top_idx)
