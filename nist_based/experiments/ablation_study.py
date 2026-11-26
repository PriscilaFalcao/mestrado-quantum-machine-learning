from __future__ import annotations

from data.preprocessing import prepare_data
from evaluation.statistical_tests import aggregate_by_key
from experiments.common import load_dataset_config, output_path, read_yaml, resolve_trigger_indices, save_json
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
    dataset_full_cfg = load_dataset_config()
    models_cfg = read_yaml("configs/models.yaml")["models"]
    exp_cfg = read_yaml("configs/experiments.yaml")["experiments"]

    dataset_cfg = dataset_full_cfg["dataset"]
    poisoning_cfg = dataset_full_cfg["poisoning"]

    seeds = [int(s) for s in exp_cfg["seeds"]]
    # Feature names do not depend on the seed, so resolve trigger indices once.
    feature_names = prepare_data(dataset_cfg, random_seed=seeds[0]).feature_names

    trigger_cfg = TriggerPoisoningConfig(
        poison_fraction=float(poisoning_cfg["poison_fraction"]),
        trigger_feature_indices=resolve_trigger_indices(poisoning_cfg, feature_names),
        trigger_value=float(poisoning_cfg["trigger_value"]),
        malicious_target_mode=str(poisoning_cfg.get("malicious_target_mode", "shift")),
        malicious_target_shift=float(poisoning_cfg.get("malicious_target_shift", 1.0)),
        malicious_target_absolute=float(poisoning_cfg.get("malicious_target_absolute", 2.0)),
    )

    tolerance = float(poisoning_cfg["tolerance"])
    selection_methods = list(exp_cfg["selection_methods"])

    rows: dict[str, dict[str, list[dict[str, float]]]] = {
        model_name: {m: [] for m in selection_methods} for model_name in ["mlp", "vqc"]
    }

    rows = {m: {a: [] for a in exp["attacks"]} for m in exp["models"]}
    for seed in seeds:
        splits = prepare_data(cfgs["dataset"], random_seed=seed).splits
        for model_name in exp["models"]:
            for attack in exp["attacks"]:
                print(f"[seed {seed}] [{model_name}] {attack}")
                rows[model_name][attack].append(run_attack(attack, model_name, splits, cfgs, seed))

    summary = {
        m: {a: aggregate_by_key(values) for a, values in by_attack.items()}
        for m, by_attack in rows.items()
    }

    out = output_path(dataset_full_cfg, "ablation_summary.json")
    save_json(out, output)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
