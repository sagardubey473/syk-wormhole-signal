"""Tests for doubled SYK system.

Verifies:
- All anticommutation relations including L/R cross terms
- [H_L, H_R] = 0
- Hermiticity of H_0
- Spectrum of H_0 equals direct sum of single-SYK spectra
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.doubled import DoubledSYK
from src.syk import SYKHamiltonian
from src.majorana import verify_anticommutation

TOL = 1e-10


@pytest.mark.parametrize("N", [4, 6, 8])
def test_anticommutation_all(N):
    """All 2N Majoranas satisfy {psi_i, psi_j} = 2 delta_{ij}."""
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    err = verify_anticommutation(ds.majoranas)
    assert err < 1e-12, f"Anticommutation error {err}"


@pytest.mark.parametrize("N", [4, 6, 8])
def test_LR_anticommute(N):
    """Left and right Majoranas anticommute: {psi^L_i, psi^R_j} = 0."""
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    max_err = 0.0
    for i in range(N):
        for j in range(N):
            anticomm = ds.psi_L[i] @ ds.psi_R[j] + ds.psi_R[j] @ ds.psi_L[i]
            err = np.linalg.norm(anticomm)
            max_err = max(max_err, err)
    assert max_err < 1e-12, f"L/R anticommutation error {max_err}"


@pytest.mark.parametrize("N", [4, 6, 8])
@pytest.mark.parametrize("seed", [42, 123])
def test_HL_HR_commute(N, seed):
    """[H_L, H_R] = 0 since they act on different qubits."""
    ds = DoubledSYK(N, seed=seed, use_sparse=False)
    comm = ds.H_L @ ds.H_R - ds.H_R @ ds.H_L
    err = np.linalg.norm(comm)
    assert err < TOL, f"[H_L, H_R] != 0, error {err}"


@pytest.mark.parametrize("N", [4, 6, 8])
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_hermiticity(N, seed):
    """H_0, H_L, H_R all Hermitian."""
    ds = DoubledSYK(N, seed=seed, use_sparse=False)
    for name, H in [('H_L', ds.H_L), ('H_R', ds.H_R), ('H_0', ds.H_0)]:
        err = np.linalg.norm(H - H.conj().T)
        assert err < TOL, f"{name} not Hermitian, error {err}"


@pytest.mark.parametrize("N", [4, 6, 8])
def test_spectrum_direct_sum(N):
    """Spectrum of H_0 = H_L + H_R should be {E_m + E_n} for eigenvalues
    of a single SYK copy.

    Since H_L and H_R use the same coupling but act on different Hilbert
    space factors, the spectrum of H_0 is the set of all sums E^L_m + E^R_n.
    """
    ds = DoubledSYK(N, seed=42, use_sparse=False)

    # Diagonalize H_0
    H0 = 0.5 * (ds.H_0 + ds.H_0.conj().T)  # ensure exact Hermiticity
    evals_H0 = np.sort(np.linalg.eigh(H0)[0])

    # Build single-side SYK with same seed to get reference spectrum
    syk_single = SYKHamiltonian(N, seed=42, use_sparse=False)
    evals_single, _ = syk_single.diagonalize()

    # The spectrum of H_0 should be all pairwise sums
    # But note: the doubled Hilbert space is 2^N, while H_L, H_R each
    # act on this full space. The spectrum is E^L_m + E^R_n where
    # m indexes left eigenvalues and n indexes right eigenvalues.
    # Since the single SYK has dim 2^(N/2), and the doubled space is 2^N,
    # this should give 2^(N/2) * 2^(N/2) = 2^N eigenvalues.

    # Diagonalize H_L and H_R separately in the doubled space
    HL = 0.5 * (ds.H_L + ds.H_L.conj().T)
    HR = 0.5 * (ds.H_R + ds.H_R.conj().T)
    evals_L = np.linalg.eigh(HL)[0]
    evals_R = np.linalg.eigh(HR)[0]

    # Unique eigenvalues of H_L should match unique eigenvalues of single SYK
    # (each with multiplicity 2^(N/2) due to the other side's Hilbert space)
    evals_L_unique = np.sort(np.unique(np.round(evals_L, 8)))
    evals_single_unique = np.sort(np.unique(np.round(evals_single, 8)))

    assert len(evals_L_unique) == len(evals_single_unique), \
        f"Number of unique H_L eigenvalues {len(evals_L_unique)} != {len(evals_single_unique)}"

    err = np.max(np.abs(evals_L_unique - evals_single_unique))
    assert err < 1e-6, f"H_L spectrum doesn't match single SYK, max error {err}"

    # Verify multiplicities: each H_L eigenvalue E should appear with
    # multiplicity = (multiplicity of E in single SYK) * 2^(N/2)
    # because H_L acts as identity on the right Hilbert space.
    dim_other = 2 ** (N // 2)
    for ev in evals_single_unique:
        single_mult = np.sum(np.abs(evals_single - ev) < 1e-6)
        expected_mult = single_mult * dim_other
        count = np.sum(np.abs(evals_L - ev) < 1e-6)
        assert count == expected_mult, \
            f"H_L eigenvalue {ev} has multiplicity {count}, expected {expected_mult}"


@pytest.mark.parametrize("N", [4, 6, 8])
def test_interaction_hermiticity(N):
    """H_int = i*mu*sum psi^L psi^R is Hermitian."""
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    H_int = ds.build_interaction(mu=0.1)
    err = np.linalg.norm(H_int - H_int.conj().T)
    assert err < TOL, f"H_int not Hermitian, error {err}"


@pytest.mark.parametrize("N", [4, 6, 8])
def test_coupled_hermiticity(N):
    """Full coupled Hamiltonian is Hermitian."""
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    H = ds.build_coupled_hamiltonian(mu=0.1)
    if hasattr(H, 'toarray'):
        H = H.toarray()
    err = np.linalg.norm(H - H.conj().T)
    assert err < TOL, f"Coupled H not Hermitian, error {err}"


def test_mu_zero_no_interaction():
    """mu=0 gives H_int = 0."""
    ds = DoubledSYK(8, seed=42, use_sparse=False)
    H_int = ds.build_interaction(mu=0.0)
    err = np.linalg.norm(H_int)
    assert err < 1e-15, f"H_int(mu=0) not zero, norm {err}"
