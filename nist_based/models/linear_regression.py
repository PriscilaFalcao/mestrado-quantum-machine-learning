from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from models.base_regressor import BaseRegressor


class LinearRegressionModel(BaseRegressor):
    """Ordinary least squares: the classical baseline without hidden layers."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_representation(self, X: np.ndarray) -> np.ndarray:
        # No hidden layer: the linear head acts directly on the inputs.
        return X
