from __future__ import annotations

import numpy as np


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_by_key(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}

    keys = rows[0].keys()
    return {k: summarize([float(r[k]) for r in rows]) for k in keys}
