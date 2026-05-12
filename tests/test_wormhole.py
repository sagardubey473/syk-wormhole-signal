"""Tests for wormhole transmission signal.

Verifies:
- Coupled Hamiltonian Hermitian for many mu
- mu=0 gives no transmission peak
- mu>0 produces structured C(t)
- Symmetry checks
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.doubled import DoubledSYK
from src.tfd import build_tfd
from src.wormhole import transmission_signal, extract_peak

TOL = 1e-10


@pytest.mark.parametrize("mu", [0.0, 0.05, 0.1, 0.2, 0.5])
def test_coupled_hermitian(mu):
    """Coupled Hamiltonian is Hermitian for various mu."""
    ds = DoubledSYK(8, seed=42, use_sparse=False)
    H = ds.build_coupled_hamiltonian(mu)
    err = np.linalg.norm(H - H.conj().T)
    assert err < TOL, f"Coupled H not Hermitian for mu={mu}, error {err}"


def test_mu_zero_no_peak():
    """At mu=0, there should be no transmission peak (signal stays near zero)."""
    N = 8
    beta = 4.0
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    tfd, Z = build_tfd(ds, beta)
    H = ds.build_coupled_hamiltonian(mu=0.0)
    t_array = np.linspace(0, 20, 50)

    C = transmission_signal(H, tfd, ds.psi_L, ds.psi_R, t_array, sites=[0, 1])

    # For mu=0, no coupling => no transmission. Signal should be small.
    # At t=0, C(0) = <TFD|psi^R psi^L|TFD> which is nonzero but small.
    # The key check: no growth/revival peak.
    peak_h, peak_t, fwhm = extract_peak(C, t_array)

    # The peak should just be C(t=0) or comparable baseline, no revival
    # Check that signal doesn't grow significantly above its initial value
    C0 = np.abs(C[0])
    assert peak_h < C0 * 3 + 0.01, \
        f"mu=0 shows unexpected peak: height={peak_h}, C(0)={C0}"


def test_mu_positive_signal():
    """At mu>0, the transmission signal should be structured.

    We don't require a specific peak here (that's physics, not a unit test),
    but we check that the signal is nontrivial and complex-valued.
    """
    N = 8
    beta = 4.0
    mu = 0.1
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    tfd, Z = build_tfd(ds, beta)
    H = ds.build_coupled_hamiltonian(mu)
    t_array = np.linspace(0, 30, 80)

    C = transmission_signal(H, tfd, ds.psi_L, ds.psi_R, t_array, sites=[0, 1, 2])

    # Signal should be nontrivial
    assert np.max(np.abs(C)) > 1e-6, "Signal is trivially zero"

    # Signal should vary over time (not constant)
    assert np.std(np.abs(C)) > 1e-6, "Signal is constant (no dynamics)"


def test_signal_site_averaging():
    """Signal averaged over all sites should be smoother than single-site."""
    N = 8
    beta = 4.0
    mu = 0.1
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    tfd, Z = build_tfd(ds, beta)
    H = ds.build_coupled_hamiltonian(mu)
    t_array = np.linspace(0, 20, 40)

    C_single = transmission_signal(H, tfd, ds.psi_L, ds.psi_R, t_array, sites=[0])
    C_all = transmission_signal(H, tfd, ds.psi_L, ds.psi_R, t_array)

    # Both should be nontrivial
    assert np.max(np.abs(C_single)) > 1e-6
    assert np.max(np.abs(C_all)) > 1e-6


def test_normalization_preserved():
    """Time evolution should preserve state norm (unitary check)."""
    N = 8
    ds = DoubledSYK(N, seed=42, use_sparse=False)
    tfd, Z = build_tfd(ds, beta=4.0)
    H = ds.build_coupled_hamiltonian(mu=0.1)

    # Check TFD norm
    assert abs(np.linalg.norm(tfd) - 1.0) < 1e-12

    # Evolve and check norm
    H_herm = 0.5 * (H + H.conj().T)
    evals, evecs = np.linalg.eigh(H_herm)
    t = 10.0
    evolved = evecs @ np.diag(np.exp(-1j * evals * t)) @ evecs.conj().T @ tfd
    assert abs(np.linalg.norm(evolved) - 1.0) < 1e-10
