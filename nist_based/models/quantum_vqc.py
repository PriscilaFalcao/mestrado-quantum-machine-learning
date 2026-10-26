from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA

from models.base_regressor import BaseRegressor
from models.torch_vqc_sim import VQCSimulator


try:
    from qiskit.circuit import ParameterVector, QuantumCircuit
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
        input_scale: float = 1.0,
        encoding: str = "ry",
        output_scale: float = 1.0,
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
        self.input_scale = input_scale
        self.encoding = encoding
        # The circuit output lives in [-1, 1]; the model is trained on y / output_scale and predicts
        # output_scale * f(x), so targets beyond +-1 stay reachable.
        self.output_scale = output_scale

        algorithm_globals.random_seed = random_state

        self.feature_map = self._build_feature_map(num_qubits, encoding)
        self.ansatz = RealAmplitudes(num_qubits=num_qubits, reps=ansatz_reps)

        self._sim = VQCSimulator(num_qubits, ansatz_reps, encoding=encoding)
        self._pca = PCA(n_components=num_qubits, whiten=True, random_state=random_state)
        self._model: VQR | None = None
        self._circuit_evaluations = 0
        self._objective_evaluations = 0

    @staticmethod
    def _build_feature_map(num_qubits: int, encoding: str) -> QuantumCircuit:
        if encoding == "zz":
            return ZZFeatureMap(feature_dimension=num_qubits, reps=1)
        if encoding == "ry":
            x = ParameterVector("x", num_qubits)
            circuit = QuantumCircuit(num_qubits)
            for q in range(num_qubits):
                circuit.ry(x[q], q)
            return circuit
        raise ValueError(f"Unsupported encoding: {encoding}")

    def _count_circuit_evaluations(self) -> None:
        """Count every circuit executed by the QNN (one per sample per forward call).

        The `VQR` callback is only invoked for SciPy-based optimizers, which left the counter at zero
        with COBYLA, so the QNN forward pass is wrapped instead.
        """
        qnn = self._model.neural_network
        forward = qnn.forward

        def counted(input_data, weights):
            self._objective_evaluations += 1
            self._circuit_evaluations += len(np.atleast_2d(input_data))
            return forward(input_data, weights)

        qnn.forward = counted

    def _build_optimizer(self):
        if self.optimizer_name == "SPSA":
            return SPSA(maxiter=self.optimizer_maxiter)
        return COBYLA(maxiter=self.optimizer_maxiter)

    def _project_features(self, X: np.ndarray, fit: bool) -> np.ndarray:
        # Whitened PCA (unit variance per component) times `input_scale`: raw component values
        # reach ~10, which turns the feature-map angles into near-random phases.
        Xq = self._pca.fit_transform(X) if fit else self._pca.transform(X)
        return Xq * self.input_scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xq = self._project_features(X, fit=True)
        optimizer = self._build_optimizer()
        self._circuit_evaluations = 0
        self._objective_evaluations = 0

        self._model = VQR(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=optimizer,
        )
        self._count_circuit_evaluations()
        self._model.fit(Xq, y / self.output_scale)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be trained before prediction.")
        Xq = self._project_features(X, fit=False)
        pred = self._model.predict(Xq)
        return self.output_scale * np.asarray(pred).reshape(-1)

    def get_representation(self, X: np.ndarray) -> np.ndarray:
        """Single-qubit <Z_q> expectations of the trained circuit, one column per qubit."""
        if self._model is None:
            raise RuntimeError("Model must be trained before calling get_representation.")
        Xq = torch.tensor(self._project_features(X, fit=False))
        theta = torch.tensor(np.asarray(self._model.weights), dtype=torch.float64)
        return self._sim.representation(theta, Xq).numpy()

    @property
    def circuit_evaluations(self) -> int:
        return self._circuit_evaluations

    @property
    def objective_evaluations(self) -> int:
        return self._objective_evaluations
