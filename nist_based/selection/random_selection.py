from __future__ import annotations

import numpy as np


def random_selection(n_samples: int, poison_fraction: float, random_seed: int) -> np.ndarray:
    poison_count = max(1, int(np.floor(n_samples * poison_fraction)))
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(n_samples, size=poison_count, replace=False))
