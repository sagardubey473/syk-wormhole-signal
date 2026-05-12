"""Tests for noise models."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.noise import (
    build_dephasing_ops,
    build_amplitude_damping_ops,
    build_correlated_dephasing_ops,
    evolve_lindblad,
)


def test_dephasing_ops_hermitian():
    """Dephasing operators should be Hermitian (Z is Hermitian)."""
    L_ops = build_dephasing_ops(3, gamma_phi=0.01)
    for L in L_ops:
        err = np.linalg.norm(L - L.conj().T)
        assert err < 1e-12


def test_trace_preservation():
    """Lindblad evolution should preserve trace of density matrix."""
    n_qubits = 2
    dim = 2**n_qubits
    rho0 = np.eye(dim, dtype=complex) / dim  # maximally mixed state

    # Simple Hamiltonian
    H = np.random.RandomState(42).randn(dim, dim)
    H = 0.5 * (H + H.T)  # Hermitianize
    H = H.astype(complex)

    L_ops = build_dephasing_ops(n_qubits, gamma_phi=0.1)
    t_array = np.linspace(0, 5, 20)

    rho_t = evolve_lindblad(rho0, H, L_ops, t_array)

    for idx in range(len(t_array)):
        tr = np.real(np.trace(rho_t[idx]))
        assert abs(tr - 1.0) < 1e-6, f"Trace = {tr} at t={t_array[idx]}"


def test_positivity_preservation():
    """Lindblad evolution should preserve positivity of density matrix."""
    n_qubits = 2
    dim = 2**n_qubits

    # Pure state
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    rho0 = np.outer(psi, psi.conj())

    H = np.zeros((dim, dim), dtype=complex)
    L_ops = build_dephasing_ops(n_qubits, gamma_phi=0.1)
    t_array = np.linspace(0, 5, 10)

    rho_t = evolve_lindblad(rho0, H, L_ops, t_array)

    for idx in range(len(t_array)):
        eigenvalues = np.linalg.eigvalsh(rho_t[idx])
        assert np.all(eigenvalues > -1e-8), \
            f"Negative eigenvalue at t={t_array[idx]}: {eigenvalues}"


def test_dephasing_removes_coherence():
    """Pure dephasing should destroy off-diagonal elements."""
    n_qubits = 1
    dim = 2

    psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(psi, psi.conj())

    H = np.zeros((dim, dim), dtype=complex)
    L_ops = build_dephasing_ops(n_qubits, gamma_phi=1.0)
    t_array = np.linspace(0, 10, 50)

    rho_t = evolve_lindblad(rho0, H, L_ops, t_array)

    # Off-diagonal should decay
    coherence_0 = np.abs(rho_t[0, 0, 1])
    coherence_end = np.abs(rho_t[-1, 0, 1])
    assert coherence_end < coherence_0 * 0.1, \
        f"Coherence not decaying: initial={coherence_0}, final={coherence_end}"
