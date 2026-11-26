from __future__ import annotations

import numpy as np

from attacks.bilevel import BilevelConfig, optimize_perturbations
from models.torch_surrogates import Params, Surrogate


def craft_clean_label_bilevel(
    surrogate: Surrogate,
    members: list[Params],
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_idx: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    trigger_feature_indices: list[int],
    trigger_value: float,
    target_value: float,
    cfg: BilevelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilevel poisoning with labels untouched: only the inputs of `base_idx` move (|delta|_inf <= eps).

    The trigger is not written into the training inputs; the optimisation has to make the
    perturbations teach the model to react to it at test time.
    """
    X_poisoned = optimize_perturbations(
        surrogate, members, X_train, y_train, base_idx, X_val, y_val,
        trigger_feature_indices, trigger_value, target_value, cfg,
    )
    return X_poisoned, y_train.copy()
