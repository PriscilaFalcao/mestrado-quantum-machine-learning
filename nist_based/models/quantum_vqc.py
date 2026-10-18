from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from models.base_regressor import BaseRegressor


try:
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit_machine_learning.algorithms import VQR
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_algorithms.utils import algorithm_globals
except ImportError as exc:  # pragma: no cover
    VQR_IMPORT_ERROR = exc
else:
    VQR_IMPORT_ERROR = None


class QuantumVQCRegressor(BaseRegressor):
    def __init__(
        self,
        num_qubits: int = 2,
        ansatz_reps: int = 1,
        optimizer_name: str = "COBYLA",
        optimizer_maxiter: int = 120,
        random_state: int = 42,
    ) -> None:
        if VQR_IMPORT_ERROR is not None:
            raise ImportError(
                "Qiskit Machine Learning is required for QuantumVQCRegressor."
            ) from VQR_IMPORT_ERROR

        self.num_qubits = num_qubits
        self.ansatz_reps = ansatz_reps
        self.optimizer_name = optimizer_name.upper()
        self.optimizer_maxiter = optimizer_maxiter
        self.random_state = random_state

        algorithm_globals.random_seed = random_state

        self.feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
        self.ansatz = RealAmplitudes(num_qubits=num_qubits, reps=ansatz_reps)

        self._pca = PCA(n_components=num_qubits, random_state=random_state)
        self._model: VQR | None = None
        self._circuit_evaluations = 0

    def _callback(self, _weights: np.ndarray, _objective: float) -> None:
        self._circuit_evaluations += 1

    def _build_optimizer(self):
        if self.optimizer_name == "SPSA":
            return SPSA(maxiter=self.optimizer_maxiter)
        return COBYLA(maxiter=self.optimizer_maxiter)

    def _project_features(self, X: np.ndarray, fit: bool) -> np.ndarray:
        return self._pca.fit_transform(X) if fit else self._pca.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xq = self._project_features(X, fit=True)
        optimizer = self._build_optimizer()
        self._circuit_evaluations = 0

        self._model = VQR(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=optimizer,
            callback=self._callback,
        )
        self._model.fit(Xq, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be trained before prediction.")
        Xq = self._project_features(X, fit=False)
        pred = self._model.predict(Xq)
        return np.asarray(pred).reshape(-1)

    def get_representation(self, X: np.ndarray) -> np.ndarray:
        Xq = self._project_features(X, fit=False)
        return np.concatenate([np.sin(Xq), np.cos(Xq)], axis=1)

    @property
    def circuit_evaluations(self) -> int:
        return self._circuit_evaluations
