from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from attacks.bilevel import BilevelConfig, craft_bilevel
from attacks.clean_label_bilevel import craft_clean_label_bilevel
from attacks.feature_collision import FeatureCollisionConfig, craft_feature_collision
from attacks.surrogate_training import build_surrogate, pretrain_snapshots
from attacks.trigger_poisoning import TriggerPoisoningConfig, apply_trigger_poisoning
from data.splits import DatasetSplits
from selection.random_selection import random_selection

ATTACKS = ["none", "trigger", "feature_collision", "bilevel", "clean_label_bilevel"]


@dataclass
class PoisonedData:
    X_train: np.ndarray
    y_train: np.ndarray
    poison_idx: np.ndarray
    craft_seconds: float


def trigger_config_from(poisoning_cfg: dict[str, Any]) -> TriggerPoisoningConfig:
    return TriggerPoisoningConfig(
        poison_fraction=float(poisoning_cfg["poison_fraction"]),
        trigger_feature_indices=list(poisoning_cfg["trigger_feature_indices"]),
        trigger_value=float(poisoning_cfg["trigger_value"]),
        malicious_target_mode=str(poisoning_cfg.get("malicious_target_mode", "shift")),
        malicious_target_shift=float(poisoning_cfg.get("malicious_target_shift", 1.0)),
        malicious_target_absolute=float(poisoning_cfg.get("malicious_target_absolute", 2.0)),
    )


def target_value_for(cfg: TriggerPoisoningConfig, y_train: np.ndarray) -> float:
    if cfg.malicious_target_mode == "absolute":
        return cfg.malicious_target_absolute
    return float(np.mean(y_train) + cfg.malicious_target_shift)


def _poison_count(n: int, fraction: float) -> int:
    return max(1, int(np.floor(n * fraction)))


def _label_closest_indices(y: np.ndarray, target_value: float, count: int) -> np.ndarray:
    """Clean-label bases: labels cannot change, so use samples already labelled near the target."""
    return np.sort(np.argsort(np.abs(y - target_value))[:count])


def craft_poisoned_dataset(
    attack: str,
    model_name: str,
    splits: DatasetSplits,
    trigger_cfg: TriggerPoisoningConfig,
    attacks_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
    seed: int,
) -> PoisonedData:
    """Build the poisoned training set for `attack`, crafted against a surrogate of `model_name`.

    Poisoned rows are random for the label-changing attacks (trigger, bilevel) and the samples with
    labels closest to the malicious target for the clean-label ones (feature_collision,
    clean_label_bilevel).
    """
    X, y = splits.X_train, splits.y_train
    count = _poison_count(len(X), trigger_cfg.poison_fraction)
    target = target_value_for(trigger_cfg, y)
    t0 = time.perf_counter()

    if attack == "none":
        return PoisonedData(X.copy(), y.copy(), np.array([], dtype=int), 0.0)

    if attack == "trigger":
        idx = random_selection(len(X), trigger_cfg.poison_fraction, seed)
        Xp, yp = apply_trigger_poisoning(X, y, idx, trigger_cfg)
        return PoisonedData(Xp, yp, idx, time.perf_counter() - t0)

    if attack not in ATTACKS:
        raise ValueError(f"Unsupported attack: {attack}")

    acfg = attacks_cfg[attack]
    surrogate = build_surrogate(model_name, models_cfg, X, seed)
    trigger_idx, trigger_value = trigger_cfg.trigger_feature_indices, trigger_cfg.trigger_value

    if attack == "feature_collision":
        fc = FeatureCollisionConfig(**acfg["attack"])
        snapshots = pretrain_snapshots(
            surrogate, X, y, seed, stages=[acfg["pretrain_steps"]], lr=acfg["pretrain_lr"]
        )
        idx = _label_closest_indices(y, target, count)
        X_targets = splits.X_val.copy()
        X_targets[:, trigger_idx] += trigger_value
        Xp = X.copy()
        Xp[idx] = craft_feature_collision(surrogate, snapshots[-1], X[idx], X_targets, fc)
        return PoisonedData(Xp, y.copy(), idx, time.perf_counter() - t0)

    bl = BilevelConfig(**acfg)
    members: list = []
    for k in range(bl.ensemble_seeds):
        members += pretrain_snapshots(
            surrogate, X, y, seed * 100 + k, stages=bl.ensemble_stages, lr=bl.pretrain_lr
        )

    if attack == "bilevel":
        idx = random_selection(len(X), trigger_cfg.poison_fraction, seed)
        Xp, yp = craft_bilevel(
            surrogate, members, X, y, idx, splits.X_val, splits.y_val, trigger_cfg, target, bl
        )
    else:
        idx = _label_closest_indices(y, target, count)
        Xp, yp = craft_clean_label_bilevel(
            surrogate, members, X, y, idx, splits.X_val, splits.y_val,
            trigger_idx, trigger_value, target, bl,
        )
    return PoisonedData(Xp, yp, idx, time.perf_counter() - t0)
