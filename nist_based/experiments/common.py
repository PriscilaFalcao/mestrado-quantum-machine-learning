from __future__ import annotations

import json
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
