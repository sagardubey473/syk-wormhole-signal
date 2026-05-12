"""Tests for observable computations."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.observables import level_spacing_ratio, spectral_form_factor, extract_lyapunov
from src.syk import SYKHamiltonian


def test_level_spacing_goe():
    """SYK level spacing ratio should be in GOE/GUE range."""
    r_all = []
    for seed in range(20):
        syk = SYKHamiltonian(10, seed=seed, use_sparse=False)
        evals, _ = syk.diagonalize()
        r_vals, r_mean = level_spacing_ratio(evals)
        r_all.extend(r_vals)

    r_mean = np.mean(r_all)
    assert 0.45 < r_mean < 0.65, f"<r> = {r_mean}, not in RMT range"


def test_level_spacing_poisson():
    """Poisson spacing ratio should be ~0.39.

    Use uncorrelated random eigenvalues (Poisson statistics).
    """
    rng = np.random.RandomState(42)
    r_all = []
    for _ in range(100):
        evals = np.sort(rng.exponential(1.0, size=100).cumsum())
        r_vals, r_mean = level_spacing_ratio(evals)
        r_all.extend(r_vals)

    r_mean = np.mean(r_all)
    assert 0.35 < r_mean < 0.42, f"Poisson <r> = {r_mean}, expected ~0.39"


def test_spectral_form_factor_initial():
    """SFF at t=0 should be 1.0 (|Tr(I)|^2/dim^2 = 1)."""
    syk = SYKHamiltonian(8, seed=42, use_sparse=False)
    evals, _ = syk.diagonalize()
    K = spectral_form_factor(evals, np.array([0.0]))
    assert abs(K[0] - 1.0) < 1e-10


def test_spectral_form_factor_late():
    """SFF at late times should plateau near 1/dim for chaotic systems."""
    syk = SYKHamiltonian(8, seed=42, use_sparse=False)
    evals, _ = syk.diagonalize()
    dim = len(evals)

    # Very late times (after Heisenberg time)
    t_late = np.linspace(1000, 2000, 10)
    K = spectral_form_factor(evals, t_late)

    # Should fluctuate around 1/dim
    assert np.mean(K) > 0.5 / dim, "SFF plateau too low"
    assert np.mean(K) < 5.0 / dim, "SFF plateau too high"
