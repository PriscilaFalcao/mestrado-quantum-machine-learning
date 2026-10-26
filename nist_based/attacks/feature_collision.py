from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.torch_surrogates import Params, Surrogate


@dataclass
class FeatureCollisionConfig:
    eps: float = 1.0
    steps: int = 300
    lr: float = 0.02
    beta: float = 0.05


def craft_feature_collision(
    surrogate: Surrogate,
    params: Params,
    X_base: np.ndarray,
    X_targets: np.ndarray,
    cfg: FeatureCollisionConfig,
) -> np.ndarray:
    """
    Clean-label feature collision (Shafahi et al. style, adapted to regression).

    Each poison starts at a base sample (its label is kept) and is moved, inside an L-inf ball of
    radius `eps`, so that its representation collides with that of a triggered target input:
        min_x ||rep(x) - rep(x_target + trigger)||^2 + beta * ||x - x_base||^2
    Targets are cycled over the poisons.
    """
    base = torch.tensor(X_base)
    targets = torch.tensor(X_targets)
    with torch.no_grad():
        rep_target = surrogate.representation(params, targets)
    rep_target = rep_target[torch.arange(len(base)) % len(targets)]

    delta = torch.zeros_like(base, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=cfg.lr)
    for _ in range(cfg.steps):
        opt.zero_grad()
        rep = surrogate.representation(params, base + delta)
        loss = ((rep - rep_target) ** 2).sum(dim=1).mean() + cfg.beta * (delta**2).sum(dim=1).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-cfg.eps, cfg.eps)

    return (base + delta).detach().numpy()
