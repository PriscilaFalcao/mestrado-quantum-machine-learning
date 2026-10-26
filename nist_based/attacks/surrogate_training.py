from __future__ import annotations

from typing import Any

import numpy as np
import torch

from models.torch_surrogates import MLPSurrogate, Params, Surrogate, VQCSurrogate


def build_surrogate(model_name: str, models_cfg: dict[str, Any], X_ref: np.ndarray, seed: int) -> Surrogate:
    if model_name == "mlp":
        cfg = models_cfg["mlp"]
        return MLPSurrogate(
            in_dim=X_ref.shape[1],
            hidden_layer_sizes=tuple(cfg.get("hidden_layer_sizes", [64, 32])),
            activation=cfg.get("activation", "relu"),
        )
    if model_name == "vqc":
        cfg = models_cfg["vqc"]
        return VQCSurrogate(
            X_ref,
            num_qubits=int(cfg.get("num_qubits", 2)),
            ansatz_reps=int(cfg.get("ansatz_reps", 1)),
            random_state=seed,
            input_scale=float(cfg.get("input_scale", 1.0)),
            encoding=cfg.get("encoding", "ry"),
            output_scale=float(cfg.get("output_scale", 1.0)),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _mse(surrogate: Surrogate, params: Params, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((surrogate.forward(params, X) - y) ** 2)


def pretrain_snapshots(
    surrogate: Surrogate,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    stages: list[int],
    lr: float,
) -> list[Params]:
    """Train on clean data with Adam and snapshot the weights at the given step counts.

    The snapshots form a MetaPoison-style ensemble over training stages (stage 0 = fresh init).
    """
    Xt, yt = torch.tensor(X), torch.tensor(y)
    params = surrogate.init_params(seed)
    opt = torch.optim.Adam(params, lr=lr)
    snapshots: list[Params] = []
    for step in range(max(stages) + 1):
        if step in stages:
            snapshots.append([p.detach().clone() for p in params])
        opt.zero_grad()
        _mse(surrogate, params, Xt, yt).backward()
        opt.step()
    return snapshots


def unrolled_train(
    surrogate: Surrogate,
    params: Params,
    X: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
) -> Params:
    """Differentiable full-batch gradient descent: the result stays attached to X and y."""
    theta = [p.detach().clone().requires_grad_() for p in params]
    for _ in range(steps):
        grads = torch.autograd.grad(_mse(surrogate, theta, X, y), theta, create_graph=True)
        theta = [p - lr * g for p, g in zip(theta, grads)]
    return theta
