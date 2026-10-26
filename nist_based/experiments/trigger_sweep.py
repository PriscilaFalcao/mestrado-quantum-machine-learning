from __future__ import annotations

import copy
import sys

from data.preprocessing import prepare_data
from experiments.attack_runner import load_configs, run_attack
from experiments.common import save_json


def main(seed: int, trigger_value: float, attacks: list[str]) -> None:
    """Run the `none` control plus `attacks` for both models at one seed and trigger magnitude.

    The control is rerun at every magnitude because a larger trigger also changes what the clean
    model predicts on triggered inputs. One process handles one (seed, trigger) pair so many can
    run in parallel; `experiments.aggregate_sweep` combines the JSON files.
    """
    cfgs = copy.deepcopy(load_configs())
    cfgs["poisoning"]["trigger_value"] = trigger_value
    splits = prepare_data(cfgs["dataset"], random_seed=seed).splits

    results: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in cfgs["experiments"]["models"]:
        results[model_name] = {}
        for attack in ["none", *attacks]:
            print(f"[seed={seed}] [trigger={trigger_value}] [{model_name}] {attack}", flush=True)
            results[model_name][attack] = run_attack(attack, model_name, splits, cfgs, seed)

    save_json(
        f"results/metrics/sweep/seed{seed}_trigger{trigger_value:g}.json",
        {"seed": seed, "trigger_value": trigger_value, **results},
    )


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3:] or ["trigger"])
