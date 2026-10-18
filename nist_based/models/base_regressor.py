from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRegressor(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(np.mean((pred - y) ** 2))

    @abstractmethod
    def get_representation(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
