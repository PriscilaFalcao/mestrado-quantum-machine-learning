from __future__ import annotations

from data.preprocessing import prepare_data
from evaluation.statistical_tests import aggregate_by_key
from experiments.attack_runner import load_configs, run_attack
from experiments.common import save_json


def main() -> None:
    cfgs = load_configs()
    exp = cfgs["experiments"]
    seeds = [int(s) for s in exp["seeds"]]

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
    save_json("results/metrics/ablation_summary.json", {"seeds": seeds, "raw": rows, "summary": summary})
    print("Saved: results/metrics/ablation_summary.json")


if __name__ == "__main__":
    main()
