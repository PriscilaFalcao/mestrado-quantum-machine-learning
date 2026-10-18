from __future__ import annotations

import time

import numpy as np

from attacks.trigger_poisoning import TriggerPoisoningConfig, apply_trigger_only, apply_trigger_poisoning
from data.preprocessing import prepare_data
from evaluation.backdoor_metrics import evaluate_backdoor
from evaluation.statistical_tests import aggregate_by_key
from experiments.common import read_yaml, save_json
from experiments.model_factory import build_model
from selection.influence_selection import influence_selection
from selection.random_selection import random_selection


def _selection_indices(method: str, X: np.ndarray, y: np.ndarray, poison_fraction: float, seed: int) -> np.ndarray:
    if method == "random":
        return random_selection(len(X), poison_fraction=poison_fraction, random_seed=seed)
    if method == "influence":
        return influence_selection(X, y, poison_fraction=poison_fraction, random_seed=seed)
    raise ValueError(f"Unsupported selection method: {method}")


def main() -> None:
    dataset_full_cfg = read_yaml("configs/dataset.yaml")
    models_cfg = read_yaml("configs/models.yaml")["models"]
    exp_cfg = read_yaml("configs/experiments.yaml")["experiments"]

    dataset_cfg = dataset_full_cfg["dataset"]
    poisoning_cfg = dataset_full_cfg["poisoning"]

    trigger_cfg = TriggerPoisoningConfig(
        poison_fraction=float(poisoning_cfg["poison_fraction"]),
        trigger_feature_indices=list(poisoning_cfg["trigger_feature_indices"]),
        trigger_value=float(poisoning_cfg["trigger_value"]),
        malicious_target_mode=str(poisoning_cfg.get("malicious_target_mode", "shift")),
        malicious_target_shift=float(poisoning_cfg.get("malicious_target_shift", 1.0)),
        malicious_target_absolute=float(poisoning_cfg.get("malicious_target_absolute", 2.0)),
    )

    tolerance = float(poisoning_cfg["tolerance"])
    seeds = [int(s) for s in exp_cfg["seeds"]]
    selection_methods = list(exp_cfg["selection_methods"])

    rows: dict[str, dict[str, list[dict[str, float]]]] = {
        model_name: {m: [] for m in selection_methods} for model_name in ["mlp", "vqc"]
    }

    for seed in seeds:
        prepared = prepare_data(dataset_cfg, random_seed=seed)
        splits = prepared.splits

        target_value = (
            trigger_cfg.malicious_target_absolute
            if trigger_cfg.malicious_target_mode == "absolute"
            else float(np.mean(splits.y_train) + trigger_cfg.malicious_target_shift)
        )

        for model_name in ["mlp", "vqc"]:
            for method in selection_methods:
                idx = _selection_indices(
                    method=method,
                    X=splits.X_train,
                    y=splits.y_train,
                    poison_fraction=trigger_cfg.poison_fraction,
                    seed=seed,
                )

                X_train_poisoned, y_train_poisoned = apply_trigger_poisoning(
                    splits.X_train,
                    splits.y_train,
                    selected_indices=idx,
                    cfg=trigger_cfg,
                )

                model = build_model(model_name, models_cfg, seed=seed)

                t0 = time.perf_counter()
                model.fit(X_train_poisoned, y_train_poisoned)
                train_seconds = time.perf_counter() - t0

                y_clean_pred = model.predict(splits.X_test)
                X_test_trigger = apply_trigger_only(
                    splits.X_test,
                    trigger_feature_indices=trigger_cfg.trigger_feature_indices,
                    trigger_value=trigger_cfg.trigger_value,
                )
                y_trigger_pred = model.predict(X_test_trigger)

                metrics = evaluate_backdoor(
                    y_true_clean=splits.y_test,
                    y_clean_pred=y_clean_pred,
                    y_trigger_pred=y_trigger_pred,
                    target_value=target_value,
                    tolerance=tolerance,
                )
                metrics["train_seconds"] = train_seconds
                metrics["poisoned_samples"] = float(len(idx))

                if model_name == "vqc":
                    metrics["circuit_evaluations"] = float(getattr(model, "circuit_evaluations", 0))

                rows[model_name][method].append(metrics)

    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for model_name, method_rows in rows.items():
        summary[model_name] = {}
        for method, values in method_rows.items():
            summary[model_name][method] = aggregate_by_key(values)

    output = {
        "seeds": seeds,
        "raw": rows,
        "summary": summary,
    }

    save_json("results/metrics/ablation_summary.json", output)
    print("Saved: results/metrics/ablation_summary.json")


if __name__ == "__main__":
    main()
