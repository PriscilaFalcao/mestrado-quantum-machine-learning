"""Validates `VQCSimulator` against the Qiskit circuit/QNN. Run from nist_based: python -m tests.check_vqc_sim"""
from __future__ import annotations

import numpy as np
import torch
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

from models.quantum_vqc import QuantumVQCRegressor as QuantumVQC
from models.torch_vqc_sim import VQCSimulator


def main() -> None:
    rng = np.random.default_rng(0)
    for enc, n, reps in [("zz", 2, 1), ("zz", 2, 2), ("zz", 3, 1), ("ry", 2, 2), ("ry", 4, 3)]:
        fm = QuantumVQC._build_feature_map(n, enc)
        an = RealAmplitudes(num_qubits=n, reps=reps)
        qnn = EstimatorQNN(circuit=fm.compose(an), input_params=fm.parameters, weight_params=an.parameters)
        sim = VQCSimulator(n, reps, encoding=enc)
        X = rng.normal(size=(5, n))
        w = rng.uniform(-np.pi, np.pi, size=sim.num_weights)
        ref = qnn.forward(X, w).reshape(-1)
        ours = sim.forward(torch.tensor(w), torch.tensor(X)).numpy()
        err = float(np.max(np.abs(ref - ours)))
        print(f"enc={enc} n={n} reps={reps} max|qiskit - sim| = {err:.2e}")
        assert err < 1e-6, "simulator does not match Qiskit"
    print("OK")


if __name__ == "__main__":
    main()
