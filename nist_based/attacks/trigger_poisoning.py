from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriggerPoisoningConfig:
    poison_fraction: float
    trigger_feature_indices: list[int]
    trigger_value: float
    malicious_target_mode: str = "shift"
    malicious_target_shift: float = 1.0
    malicious_target_absolute: float = 2.0


def _compute_malicious_targets(
    y_selected: np.ndarray,
    mode: str,
    shift: float,
    absolute: float,
) -> np.ndarray:
    if mode == "absolute":
        return np.full_like(y_selected, fill_value=absolute)
    return y_selected + shift


def apply_trigger_poisoning(
    X: np.ndarray,
    y: np.ndarray,
    selected_indices: np.ndarray,
    cfg: TriggerPoisoningConfig,
) -> tuple[np.ndarray, np.ndarray]:
    X_poisoned = X.copy()
    y_poisoned = y.copy()

    X_poisoned[np.ix_(selected_indices, cfg.trigger_feature_indices)] += cfg.trigger_value

    y_poisoned[selected_indices] = _compute_malicious_targets(
        y_selected=y_poisoned[selected_indices],
        mode=cfg.malicious_target_mode,
        shift=cfg.malicious_target_shift,
        absolute=cfg.malicious_target_absolute,
    )

    return X_poisoned, y_poisoned


def apply_trigger_only(X: np.ndarray, trigger_feature_indices: list[int], trigger_value: float) -> np.ndarray:
    X_trigger = X.copy()
    X_trigger[:, trigger_feature_indices] += trigger_value
    return X_trigger
