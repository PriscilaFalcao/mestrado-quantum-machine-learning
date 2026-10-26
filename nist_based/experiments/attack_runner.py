from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from attacks.pipeline import craft_poisoned_dataset, target_value_for, trigger_config_from
from attacks.trigger_poisoning import apply_trigger_only
from data.splits import DatasetSplits
from evaluation.backdoor_metrics import evaluate_backdoor
from experiments.common import read_yaml
from experiments.model_factory import build_model


def load_configs() -> dict[str, Any]:
    dataset_full = read_yaml("configs/dataset.yaml")
    return {
        "dataset": dataset_full["dataset"],
        "poisoning": dataset_full["poisoning"],
        "models": read_yaml("configs/models.yaml")["models"],
        "attacks": read_yaml("configs/attacks.yaml")["attacks"],
        "experiments": read_yaml("configs/experiments.yaml")["experiments"],
    }


def splits_for_model(splits: DatasetSplits, model_name: str, models_cfg: dict[str, Any], seed: int) -> DatasetSplits:
    """Random training subset when the model config sets `max_train_samples` (used for the VQC).

    The subset is drawn before any poisoning, so the poison fraction applies to the subset and the
    validation/test splits stay identical across models.
    """
    limit = models_cfg[model_name].get("max_train_samples")
    if not limit or len(splits.X_train) <= int(limit):
        return splits
    keep = np.sort(np.random.default_rng(seed).choice(len(splits.X_train), size=int(limit), replace=False))
    return replace(splits, X_train=splits.X_train[keep], y_train=splits.y_train[keep])


def run_attack(attack: str, model_name: str, splits: DatasetSplits, cfgs: dict[str, Any], seed: int) -> dict[str, float]:
    """Craft the poisoned set, train the real model on it and measure the backdoor."""
    splits = splits_for_model(splits, model_name, cfgs["models"], seed)
    trigger_cfg = trigger_config_from(cfgs["poisoning"])
    poisoned = craft_poisoned_dataset(
        attack, model_name, splits, trigger_cfg, cfgs["attacks"], cfgs["models"], seed
    )

    model = build_model(model_name, cfgs["models"], seed=seed)
    t0 = time.perf_counter()
    model.fit(poisoned.X_train, poisoned.y_train)
    train_seconds = time.perf_counter() - t0

    X_test_trigger = apply_trigger_only(
        splits.X_test, trigger_cfg.trigger_feature_indices, trigger_cfg.trigger_value
    )
    metrics = evaluate_backdoor(
        y_true_clean=splits.y_test,
        y_clean_pred=model.predict(splits.X_test),
        y_trigger_pred=model.predict(X_test_trigger),
        target_value=target_value_for(trigger_cfg, splits.y_train),
        tolerance=float(cfgs["poisoning"]["tolerance"]),
    )
    shift = np.abs(poisoned.X_train - splits.X_train)
    metrics.update(
        craft_seconds=poisoned.craft_seconds,
        train_seconds=train_seconds,
        poisoned_samples=float(len(poisoned.poison_idx)),
        mean_linf_perturbation=float(shift.max(axis=1)[poisoned.poison_idx].mean()) if len(poisoned.poison_idx) else 0.0,
    )
    if model_name == "vqc":
        metrics["circuit_evaluations"] = float(model.circuit_evaluations)
        metrics["objective_evaluations"] = float(model.objective_evaluations)
    return metrics
