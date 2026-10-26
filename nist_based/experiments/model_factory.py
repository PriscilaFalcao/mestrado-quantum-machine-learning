from __future__ import annotations

from typing import Any

from models.classical_mlp import ClassicalMLPRegressor
from models.quantum_vqc import QuantumVQCRegressor


def build_model(model_name: str, models_cfg: dict[str, Any], seed: int):
    if model_name == "mlp":
        cfg = models_cfg["mlp"]
        return ClassicalMLPRegressor(
            hidden_layer_sizes=tuple(cfg.get("hidden_layer_sizes", [64, 32])),
            activation=cfg.get("activation", "relu"),
            alpha=float(cfg.get("alpha", 1e-4)),
            learning_rate_init=float(cfg.get("learning_rate_init", 1e-3)),
            max_iter=int(cfg.get("max_iter", 800)),
            random_state=seed,
        )

    if model_name == "vqc":
        cfg = models_cfg["vqc"]
        return QuantumVQCRegressor(
            num_qubits=int(cfg.get("num_qubits", 2)),
            ansatz_reps=int(cfg.get("ansatz_reps", 1)),
            optimizer_name=cfg.get("optimizer", "COBYLA"),
            optimizer_maxiter=int(cfg.get("optimizer_maxiter", 120)),
            random_state=seed,
            input_scale=float(cfg.get("input_scale", 1.0)),
            encoding=cfg.get("encoding", "ry"),
            output_scale=float(cfg.get("output_scale", 1.0)),
        )

    raise ValueError(f"Unsupported model: {model_name}")
