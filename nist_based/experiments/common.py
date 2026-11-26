from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_yaml(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(relative_dir: str) -> Path:
    out_dir = PROJECT_ROOT / relative_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(relative_path: str, payload: dict[str, Any]) -> None:
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_OUTPUT_DIR = "results/metrics"


def load_dataset_config() -> dict[str, Any]:
    """Dataset config path from `--dataset-config`, env `DATASET_CONFIG`, or the default file."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-config", default=None)
    args, _ = parser.parse_known_args()
    path = args.dataset_config or os.environ.get("DATASET_CONFIG") or DEFAULT_DATASET_CONFIG
    print(f"Using dataset config: {path}")
    return read_yaml(path)


def output_path(dataset_full_cfg: dict[str, Any], filename: str) -> str:
    out_dir = dataset_full_cfg["dataset"].get("output_dir", DEFAULT_OUTPUT_DIR)
    return f"{out_dir}/{filename}"


def resolve_trigger_indices(poisoning_cfg: dict[str, Any], feature_names: list[str]) -> list[int]:
    """Use `trigger_feature_names` when given (robust to one-hot expansion), else `trigger_feature_indices`."""
    names = poisoning_cfg.get("trigger_feature_names")
    if names:
        missing = [n for n in names if n not in feature_names]
        if missing:
            raise ValueError(f"trigger_feature_names not found in features: {missing}. Available: {feature_names}")
        return [feature_names.index(n) for n in names]

    indices = [int(i) for i in poisoning_cfg["trigger_feature_indices"]]
    out_of_range = [i for i in indices if not -len(feature_names) <= i < len(feature_names)]
    if out_of_range:
        raise ValueError(f"trigger_feature_indices {out_of_range} out of range for {len(feature_names)} features.")
    return indices
