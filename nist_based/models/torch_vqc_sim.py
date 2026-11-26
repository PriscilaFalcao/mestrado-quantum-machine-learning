from __future__ import annotations

import itertools
import math

import numpy as np
import torch


class VQCSimulator:
    """
    Differentiable statevector simulator of the circuit used by `QuantumVQCRegressor`:
    `ZZFeatureMap` (full entanglement) followed by `RealAmplitudes` (reverse_linear), read out
    with the default `EstimatorQNN` observable Z^{⊗n}. Qubit 0 is the least significant bit
    (Qiskit convention). It is validated numerically against Qiskit in `tests/check_vqc_sim.py`.

    It exists so that attacks can differentiate through the quantum model (w.r.t. the inputs and
    through unrolled training), which Qiskit's COBYLA-driven `VQR` does not allow.
    """

    def __init__(self, num_qubits: int, ansatz_reps: int, feature_reps: int = 1, encoding: str = "zz") -> None:
        self.encoding = encoding
        self.n = num_qubits
        self.ansatz_reps = ansatz_reps
        self.feature_reps = feature_reps
        self.dim = 2**num_qubits
        self.num_weights = num_qubits * (ansatz_reps + 1)

        ks = np.arange(self.dim)
        self.bits = torch.tensor(
            [[(k >> q) & 1 for q in range(num_qubits)] for k in ks], dtype=torch.float64
        )
        self.pairs = list(itertools.combinations(range(num_qubits), 2))

        h = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
        walsh = np.array([[1.0]])
        for _ in range(num_qubits):
            walsh = np.kron(h, walsh)
        self.walsh = torch.tensor(walsh, dtype=torch.complex128)

        self.cx_perms = [self._cx_perm(c, t) for c, t in self._entangling_pairs()]
        self.parity_sign = torch.tensor([(-1.0) ** bin(k).count("1") for k in ks], dtype=torch.float64)
        self.z_signs = 1.0 - 2.0 * self.bits  # (dim, n): eigenvalue of Z_q on basis state k

    def _entangling_pairs(self) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(self.n - 2, -1, -1)]

    def _cx_perm(self, control: int, target: int) -> torch.Tensor:
        perm = []
        for k in range(self.dim):
            perm.append(k ^ (1 << target) if (k >> control) & 1 else k)
        return torch.tensor(perm, dtype=torch.long)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding == "ry":
            # Angle encoding: RY(x_q) on each qubit of |0...0>, repeated `feature_reps` times.
            state = torch.zeros(x.shape[0], self.dim, dtype=torch.complex128)
            state[:, 0] = 1.0
            for _ in range(self.feature_reps):
                for q in range(self.n):
                    state = self._ry_layer_single(state, q, x[:, q])
            return state
        batch = x.shape[0]
        state = torch.zeros(batch, self.dim, dtype=torch.complex128)
        state[:, 0] = 1.0
        bits = self.bits
        for _ in range(self.feature_reps):
            state = state @ self.walsh.T
            phase = 2.0 * x @ bits.T  # P(2 x_q) on each qubit
            for i, j in self.pairs:
                coef = 2.0 * (math.pi - x[:, i]) * (math.pi - x[:, j])
                parity = (bits[:, i] + bits[:, j]) % 2.0
                phase = phase + coef[:, None] * parity[None, :]
            state = state * torch.exp(1j * phase)
        return state

    def _ry_layer(self, state: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        psi = state.reshape((batch,) + (2,) * self.n)
        for q in range(self.n):
            c, s = torch.cos(angles[q] / 2.0), torch.sin(angles[q] / 2.0)
            gate = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex128)
            axis = self.n - q  # axis 0 is the batch; axes 1..n are qubits n-1..0
            psi = torch.movedim(torch.movedim(psi, axis, -1) @ gate.T, -1, axis)
        return psi.reshape(batch, self.dim)

    def _ry_layer_single(self, state: torch.Tensor, q: int, angle: torch.Tensor) -> torch.Tensor:
        """RY on qubit q with a per-sample angle (shape (B,))."""
        batch = state.shape[0]
        psi = state.reshape((batch,) + (2,) * self.n)
        axis = self.n - q
        c, s = torch.cos(angle / 2.0), torch.sin(angle / 2.0)
        gate = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2).to(torch.complex128)  # (B,2,2)
        moved = torch.movedim(psi, axis, -1)  # (B, ..., 2)
        shape = (batch,) + (1,) * (self.n - 1) + (2, 2)
        out = torch.einsum("b...j,bij->b...i", moved, gate)
        return torch.movedim(out, -1, axis).reshape(batch, self.dim)

    def _ansatz(self, state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        for r in range(self.ansatz_reps):
            state = self._ry_layer(state, theta[r * self.n : (r + 1) * self.n])
            for perm in self.cx_perms:
                state = state[:, torch.argsort(perm)]
        return self._ry_layer(state, theta[self.ansatz_reps * self.n :])

    def state(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._ansatz(self._feature_map(x), theta)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """<Z^{⊗n}> per sample, shape (B,)."""
        prob = self.state(theta, x).abs() ** 2
        return prob @ self.parity_sign

    def representation(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Vector of single-qubit <Z_q> expectations per sample, shape (B, n)."""
        prob = self.state(theta, x).abs() ** 2
        return prob @ self.z_signs
