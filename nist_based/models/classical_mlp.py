from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

from models.base_regressor import BaseRegressor


class ClassicalMLPRegressor(BaseRegressor):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 800,
        random_state: int = 42,
    ) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.activation = activation

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0.0, x)
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "logistic":
            return 1.0 / (1.0 + np.exp(-x))
        return x

    def get_representation(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "coefs_"):
            raise RuntimeError("Model must be trained before calling get_representation.")

        h = X
        last_hidden_layer = len(self.model.coefs_) - 1
        for i in range(last_hidden_layer):
            h = h @ self.model.coefs_[i] + self.model.intercepts_[i]
            h = self._apply_activation(h)

        return h
