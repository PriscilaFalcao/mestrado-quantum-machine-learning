from __future__ import annotations

from data.preprocessing import prepare_data
from experiments.attack_runner import load_configs, run_attack
from experiments.common import save_json


def main() -> None:
    cfgs = load_configs()
    seed = int(cfgs["dataset"]["random_seed"])
    splits = prepare_data(cfgs["dataset"], random_seed=seed).splits

    results: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in cfgs["experiments"]["models"]:
        results[model_name] = {}
        for attack in cfgs["experiments"]["attacks"]:
            print(f"[{model_name}] {attack}")
            results[model_name][attack] = run_attack(attack, model_name, splits, cfgs, seed)

    save_json("results/metrics/attack_comparison.json", results)
    print("Saved: results/metrics/attack_comparison.json")


if __name__ == "__main__":
    main()
