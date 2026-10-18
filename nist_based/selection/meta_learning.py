from __future__ import annotations

import numpy as np


def meta_learning_inspired_selection(
    X: np.ndarray,
    y: np.ndarray,
    poison_fraction: float,
    random_seed: int,
) -> np.ndarray:
    """
    Placeholder selection inspired by consistency across perturbed views.
    This is intentionally lightweight and not the original MetaPoison.
    """
    poison_count = max(1, int(np.floor(len(X) * poison_fraction)))

    rng = np.random.default_rng(random_seed)
    scores = np.zeros(len(X), dtype=float)

    for _ in range(3):
        noise = rng.normal(0.0, 0.05, size=X.shape)
        X_noisy = X + noise
        centered = X_noisy - X_noisy.mean(axis=0, keepdims=True)
        variance_score = np.linalg.norm(centered, axis=1)
        target_score = np.abs(y - np.median(y))
        scores += variance_score * target_score

    top_idx = np.argsort(scores)[-poison_count:]
    return np.sort(top_idx)
