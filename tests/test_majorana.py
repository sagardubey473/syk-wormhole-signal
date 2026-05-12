"""Tests for Majorana fermion operators.

Verifies:
- Anticommutation relations {psi_i, psi_j} = 2 delta_{ij}
- Hermiticity of each operator
- psi_i^2 = I
- Sparse vs dense consistency
"""

import numpy as np
import pytest
from scipy import sparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.majorana import (
    build_majorana_operators,
    verify_anticommutation,
    verify_hermiticity,
)

TOL = 1e-12


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12, 14])
def test_anticommutation_dense(N):
    """All anticommutation relations hold for dense operators."""
    ops = build_majorana_operators(N, use_sparse=False)
    err = verify_anticommutation(ops)
    assert err < TOL, f"Anticommutation error {err} for N={N}"


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_anticommutation_sparse(N):
    """All anticommutation relations hold for sparse operators."""
    ops = build_majorana_operators(N, use_sparse=True)
    err = verify_anticommutation(ops)
    assert err < TOL, f"Anticommutation error {err} for N={N}"


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12, 14])
def test_hermiticity(N):
    """All Majorana operators are Hermitian."""
    ops = build_majorana_operators(N, use_sparse=False)
    err = verify_hermiticity(ops)
    assert err < TOL, f"Hermiticity error {err} for N={N}"


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12, 14])
def test_square_is_identity(N):
    """psi_i^2 = I for all i."""
    ops = build_majorana_operators(N, use_sparse=False)
    dim = ops[0].shape[0]
    identity = np.eye(dim)
    for i, op in enumerate(ops):
        diff = op @ op - identity
        err = np.linalg.norm(diff)
        assert err < TOL, f"psi_{i}^2 != I, error {err} for N={N}"


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_sparse_dense_consistency(N):
    """Sparse and dense representations produce the same operators."""
    dense_ops = build_majorana_operators(N, use_sparse=False)
    sparse_ops = build_majorana_operators(N, use_sparse=True)
    for i in range(N):
        diff = dense_ops[i] - sparse_ops[i].toarray()
        err = np.linalg.norm(diff)
        assert err < TOL, f"Sparse/dense mismatch for psi_{i}, N={N}"


def test_correct_dimension():
    """Operators have correct Hilbert space dimension 2^(N/2)."""
    for N in [4, 6, 8, 10]:
        ops = build_majorana_operators(N, use_sparse=False)
        expected_dim = 2 ** (N // 2)
        assert len(ops) == N
        for op in ops:
            assert op.shape == (expected_dim, expected_dim)


def test_invalid_N():
    """Raise ValueError for invalid N."""
    with pytest.raises(ValueError):
        build_majorana_operators(3)
    with pytest.raises(ValueError):
        build_majorana_operators(1)
    with pytest.raises(ValueError):
        build_majorana_operators(0)


def test_traceless():
    """Majorana operators should be traceless for N >= 4."""
    for N in [4, 6, 8, 10]:
        ops = build_majorana_operators(N, use_sparse=False)
        for i, op in enumerate(ops):
            tr = np.abs(np.trace(op))
            assert tr < TOL, f"psi_{i} not traceless, Tr={tr} for N={N}"
