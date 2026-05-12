"""Tests for TFD state construction.

Verifies:
- Norm = 1 within 1e-12
- Partial trace over R equals exp(-beta*H_L)/Z
- At beta=0, reduced state is I/dim
- TFD energy expectation matches 2 * thermal energy
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.doubled import DoubledSYK
from src.syk import SYKHamiltonian
from src.tfd import build_tfd, partial_trace_R, entanglement_entropy

TOL = 1e-10


@pytest.mark.parametrize("N", [4, 6, 8])
@pytest.mark.parametrize("beta", [0.0, 1.0, 4.0, 8.0])
def test_normalization(N, beta):
    """TFD state has unit norm."""
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    tfd, Z = build_tfd(ds, beta)
    norm = np.linalg.norm(tfd)
    assert abs(norm - 1.0) < 1e-12, f"TFD norm = {norm}, expected 1.0"


@pytest.mark.parametrize("N", [4, 6, 8])
@pytest.mark.parametrize("beta", [1.0, 4.0, 8.0])
def test_partial_trace_thermal(N, beta):
    """Partial trace over R gives thermal state exp(-beta*H_L)/Z.

    rho_L = Tr_R(|TFD><TFD|) should equal exp(-beta*H_single)/Z
    in the computational basis, where H_single is the single-copy SYK.
    """
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    tfd, Z = build_tfd(ds, beta)

    dim_single = 2 ** (N // 2)
    rho_L = partial_trace_R(tfd, dim_single, dim_single)

    # Build expected thermal state from single SYK
    syk = SYKHamiltonian(N, seed=42, use_sparse=False)
    rho_thermal, Z_check = syk.thermal_density_matrix(beta)

    # Compare
    err = np.linalg.norm(rho_L - rho_thermal, 'fro')
    assert err < TOL, f"Partial trace error {err} for N={N}, beta={beta}"


@pytest.mark.parametrize("N", [4, 6, 8])
def test_beta_zero_maximally_mixed(N):
    """At beta=0, reduced state is I/dim (maximally entangled)."""
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    tfd, Z = build_tfd(ds, beta=0.0)

    dim_single = 2 ** (N // 2)
    rho_L = partial_trace_R(tfd, dim_single, dim_single)

    # Should be I/dim
    expected = np.eye(dim_single) / dim_single
    err = np.linalg.norm(rho_L - expected, 'fro')
    assert err < TOL, f"beta=0 reduced state error {err}"


@pytest.mark.parametrize("N", [4, 6, 8])
def test_energy_expectation(N):
    """<TFD|H_0|TFD> should equal 2 * thermal energy.

    <H_0> = <H_L> + <H_R> = 2 * Tr(rho_thermal * H_single)
    """
    beta = 4.0
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    tfd, Z = build_tfd(ds, beta)

    # <H_0> from TFD
    H_0 = ds.get_H_0_dense()
    E_tfd = np.real(tfd.conj() @ H_0 @ tfd)

    # 2 * thermal energy from single SYK
    syk = SYKHamiltonian(N, seed=42, use_sparse=False)
    rho, _ = syk.thermal_density_matrix(beta)
    H_single = syk.get_hamiltonian_dense()
    E_thermal = np.real(np.trace(rho @ H_single))

    err = abs(E_tfd - 2 * E_thermal)
    assert err < TOL, f"Energy mismatch: <H_0>_TFD={E_tfd:.6f}, 2*E_thermal={2*E_thermal:.6f}"


@pytest.mark.parametrize("N", [4, 6, 8])
def test_entanglement_entropy_beta0(N):
    """At beta=0, entanglement entropy = log(dim_single)."""
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    tfd, Z = build_tfd(ds, beta=0.0)

    dim_single = 2 ** (N // 2)
    S = entanglement_entropy(tfd, dim_single, dim_single)
    S_max = np.log(dim_single)

    assert abs(S - S_max) < 1e-10, f"S = {S:.6f}, expected {S_max:.6f}"


@pytest.mark.parametrize("N", [4, 6, 8])
def test_entanglement_decreases_with_beta(N):
    """Entanglement entropy should decrease with increasing beta.

    As temperature decreases (beta increases), the TFD approaches a product
    state (ground state x ground state), so S -> 0.
    """
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    dim_single = 2 ** (N // 2)

    entropies = []
    for beta in [0.0, 2.0, 8.0, 32.0]:
        tfd, Z = build_tfd(ds, beta)
        S = entanglement_entropy(tfd, dim_single, dim_single)
        entropies.append(S)

    # Should be monotonically decreasing (or at least non-increasing)
    for i in range(len(entropies) - 1):
        assert entropies[i] >= entropies[i+1] - 1e-10, \
            f"Entropy not decreasing: S(beta[{i}])={entropies[i]:.4f} < S(beta[{i+1}])={entropies[i+1]:.4f}"
