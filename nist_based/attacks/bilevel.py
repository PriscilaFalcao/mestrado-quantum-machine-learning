from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from attacks.surrogate_training import unrolled_train
from attacks.trigger_poisoning import TriggerPoisoningConfig, apply_trigger_poisoning
from models.torch_surrogates import Params, Surrogate


@dataclass
class BilevelConfig:
    eps: float = 0.5
    outer_steps: int = 60
    outer_lr: float = 0.05
    inner_steps: int = 10
    inner_lr: float = 0.05
    clean_weight: float = 0.5
    pretrain_lr: float = 0.01
    ensemble_stages: list[int] = field(default_factory=lambda: [0, 25, 100])
    ensemble_seeds: int = 2


def optimize_perturbations(
    surrogate: Surrogate,
    members: list[Params],
    X_start: np.ndarray,
    y_start: np.ndarray,
    poison_idx: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    trigger_feature_indices: list[int],
    trigger_value: float,
    target_value: float,
    cfg: BilevelConfig,
) -> np.ndarray:
    """
    MetaPoison-style bilevel optimisation of the poison inputs.

    Inner problem: starting from each ensemble member (surrogate snapshots at different training
    stages / inits), run `inner_steps` of differentiable gradient descent on the poisoned set.
    Outer problem: choose the perturbation delta (|delta|_inf <= eps) on the poisoned rows that
    minimises, through the unrolled training,
        MSE(f(X_val + trigger), target) + clean_weight * MSE(f(X_val), y_val).
    `X_start`/`y_start` is the training set before the perturbation (labels are never altered here,
    so clean-label vs dirty-label is decided by how the caller built `y_start`).
    """
    X0, y0 = torch.tensor(X_start), torch.tensor(y_start)
    Xv, yv = torch.tensor(X_val), torch.tensor(y_val)
    Xv_trigger = Xv.clone()
    Xv_trigger[:, trigger_feature_indices] += trigger_value
    idx = torch.tensor(poison_idx, dtype=torch.long)

    delta = torch.zeros(len(poison_idx), X0.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=cfg.outer_lr)

    for _ in range(cfg.outer_steps):
        opt.zero_grad()
        X_poisoned = X0.clone()
        X_poisoned[idx] = X0[idx] + delta

        total = torch.zeros((), dtype=torch.float64)
        for member in members:
            theta = unrolled_train(surrogate, member, X_poisoned, y0, cfg.inner_steps, cfg.inner_lr)
            attack_loss = torch.mean((surrogate.forward(theta, Xv_trigger) - target_value) ** 2)
            clean_loss = torch.mean((surrogate.forward(theta, Xv) - yv) ** 2)
            total = total + attack_loss + cfg.clean_weight * clean_loss
        (total / len(members)).backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-cfg.eps, cfg.eps)

    X_final = X0.clone()
    X_final[idx] = X0[idx] + delta.detach()
    return X_final.numpy()


def craft_bilevel(
    surrogate: Surrogate,
    members: list[Params],
    X_train: np.ndarray,
    y_train: np.ndarray,
    poison_idx: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    trigger_cfg: TriggerPoisoningConfig,
    target_value: float,
    cfg: BilevelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Dirty-label bilevel poisoning: trigger + malicious labels on `poison_idx`, then the
    inputs of those rows are refined by `optimize_perturbations` (|delta|_inf <= eps)."""
    X_start, y_start = apply_trigger_poisoning(X_train, y_train, poison_idx, trigger_cfg)
    X_poisoned = optimize_perturbations(
        surrogate, members, X_start, y_start, poison_idx, X_val, y_val,
        trigger_cfg.trigger_feature_indices, trigger_cfg.trigger_value, target_value, cfg,
    )
    return X_poisoned, y_start
