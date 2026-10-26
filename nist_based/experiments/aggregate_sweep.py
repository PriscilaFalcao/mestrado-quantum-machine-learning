from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

from experiments.common import PROJECT_ROOT, save_json

METRICS = ["backdoor_success_rate", "prediction_shift", "clean_mae", "trigger_mae"]
ATTACK_ORDER = ["none", "trigger", "feature_collision", "bilevel", "clean_label_bilevel"]


def _ms(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.3f} ± {(arr.std(ddof=1) if len(arr) > 1 else 0.0):.3f}"


def main() -> None:
    runs = [json.load(open(f)) for f in sorted(glob.glob(str(PROJECT_ROOT / "results/metrics/sweep/*.json")))]
    by_key: dict[tuple[str, float, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for run in runs:
        for model in ("mlp", "vqc"):
            for attack, metrics in run[model].items():
                by_key[(model, run["trigger_value"], attack)][run["seed"]] = metrics

    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    lines = [
        "| Modelo | Gatilho | Ataque | Sucesso do backdoor | Ganho sobre `none` | Deslocamento | MAE limpo | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in ("mlp", "vqc"):
        for trig in sorted({k[1] for k in by_key if k[0] == model}):
            for attack in ATTACK_ORDER:
                rows = by_key.get((model, trig, attack))
                if not rows:
                    continue
                control = by_key[(model, trig, "none")]
                seeds = sorted(rows)
                bsr = [rows[s]["backdoor_success_rate"] for s in seeds]
                gain = [rows[s]["backdoor_success_rate"] - control[s]["backdoor_success_rate"] for s in seeds if s in control]
                summary.setdefault(model, {}).setdefault(str(trig), {})[attack] = {
                    "n_seeds": len(seeds),
                    **{f"{m}_mean": float(np.mean([rows[s][m] for s in seeds])) for m in METRICS},
                    **{f"{m}_std": float(np.std([rows[s][m] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0 for m in METRICS},
                    "bsr_gain_mean": float(np.mean(gain)),
                    "bsr_gain_std": float(np.std(gain, ddof=1)) if len(gain) > 1 else 0.0,
                }
                lines.append(
                    f"| {model.upper()} | {trig:g} | {attack} | {_ms(bsr)} | "
                    f"{'–' if attack == 'none' else _ms(gain)} | "
                    f"{_ms([rows[s]['prediction_shift'] for s in seeds])} | "
                    f"{_ms([rows[s]['clean_mae'] for s in seeds])} | {len(seeds)} |"
                )

    save_json("results/metrics/sweep_summary.json", summary)
    (PROJECT_ROOT / "results/metrics/sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
