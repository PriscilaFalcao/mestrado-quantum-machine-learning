from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import torch
from sklearn.decomposition import PCA

from models.torch_vqc_sim import VQCSimulator

Params = list[torch.Tensor]


class Surrogate(Protocol):
    """Functional, differentiable stand-in for a model, used to craft attacks."""

    def init_params(self, seed: int) -> Params: ...

    def forward(self, params: Params, X: torch.Tensor) -> torch.Tensor: ...

    def representation(self, params: Params, X: torch.Tensor) -> torch.Tensor: ...


class MLPSurrogate:
    def __init__(self, in_dim: int, hidden_layer_sizes: tuple[int, ...], activation: str = "relu") -> None:
        self.sizes = [in_dim, *hidden_layer_sizes, 1]
        self.activation = {"relu": torch.relu, "tanh": torch.tanh, "logistic": torch.sigmoid}.get(
            activation, lambda x: x
        )

    def init_params(self, seed: int) -> Params:
        gen = torch.Generator().manual_seed(seed)
        params: Params = []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            w = (torch.rand(fan_in, fan_out, generator=gen, dtype=torch.float64) * 2 - 1) * bound
            params += [w.requires_grad_(), torch.zeros(fan_out, dtype=torch.float64, requires_grad=True)]
        return params

    def representation(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        h = X
        for w, b in zip(params[:-2:2], params[1:-2:2]):
            h = self.activation(h @ w + b)
        return h

    def forward(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        return (self.representation(params, X) @ params[-2] + params[-1]).reshape(-1)


class VQCSurrogate:
    """Quantum surrogate: fixed linear PCA projection followed by the differentiable VQC simulator."""

    def __init__(
        self, X_ref: np.ndarray, num_qubits: int, ansatz_reps: int, random_state: int = 42, input_scale: float = 1.0,
        encoding: str = "ry",
        output_scale: float = 1.0,
    ) -> None:
        self.output_scale = output_scale
        pca = PCA(n_components=num_qubits, whiten=True, random_state=random_state).fit(X_ref)
        self.mean = torch.tensor(pca.mean_, dtype=torch.float64)
        scale = input_scale / np.sqrt(pca.explained_variance_)
        self.components = torch.tensor(pca.components_ * scale[:, None], dtype=torch.float64)
        self.sim = VQCSimulator(num_qubits, ansatz_reps, encoding=encoding)

    def init_params(self, seed: int) -> Params:
        gen = torch.Generator().manual_seed(seed)
        theta = torch.randn(self.sim.num_weights, generator=gen, dtype=torch.float64)
        return [theta.requires_grad_()]

    def _project(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) @ self.components.T

    def forward(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        return self.output_scale * self.sim.forward(params[0], self._project(X))

    def representation(self, params: Params, X: torch.Tensor) -> torch.Tensor:
        return self.sim.representation(params[0], self._project(X))
