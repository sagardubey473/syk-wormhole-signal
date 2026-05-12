"""Majorana fermion operators via Jordan-Wigner transformation.

Constructs N Majorana operators satisfying {psi_i, psi_j} = 2 delta_{ij}
using N/2 qubits. See CONVENTIONS.md sections 1-2.

Jordan-Wigner mapping (1-indexed qubits, 0-indexed Majoranas):
    psi_{2k}   = (Z_1 x Z_2 x ... x Z_k) x X_{k+1}     k = 0, 1, ..., N/2 - 1
    psi_{2k+1} = (Z_1 x Z_2 x ... x Z_k) x Y_{k+1}     k = 0, 1, ..., N/2 - 1

where X, Y, Z are Pauli matrices and x denotes tensor product.
"""

import numpy as np
from scipy import sparse
from functools import lru_cache


# Pauli matrices (dense)
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Pauli matrices (sparse)
_I2_sp = sparse.eye(2, format='csr', dtype=complex)
_X_sp = sparse.csr_matrix(_X)
_Y_sp = sparse.csr_matrix(_Y)
_Z_sp = sparse.csr_matrix(_Z)


def _kron_list(matrices):
    """Tensor product of a list of matrices (dense)."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def _kron_list_sparse(matrices):
    """Tensor product of a list of sparse matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = sparse.kron(result, m, format='csr')
    return result


def build_majorana_operators(N, use_sparse=None):
    """Build N Majorana operators on N/2 qubits.

    Parameters
    ----------
    N : int
        Number of Majorana fermions. Must be even and >= 2.
    use_sparse : bool or None
        If None, auto-select: sparse for N > 12, dense otherwise.

    Returns
    -------
    list of ndarray or sparse matrix
        N Majorana operators, each of dimension 2^(N/2) x 2^(N/2).
    """
    if N < 2 or N % 2 != 0:
        raise ValueError(f"N must be even and >= 2, got {N}")

    n_qubits = N // 2
    if use_sparse is None:
        use_sparse = n_qubits > 6  # sparse for > 6 qubits (N > 12)

    if use_sparse:
        return _build_majorana_sparse(N, n_qubits)
    else:
        return _build_majorana_dense(N, n_qubits)


def _build_majorana_dense(N, n_qubits):
    """Build dense Majorana operators."""
    ops = []
    for i in range(N):
        k = i // 2      # qubit index (0-based)
        is_y = i % 2     # 0 -> X (even Majorana), 1 -> Y (odd Majorana)

        factors = []
        # Jordan-Wigner string: Z on qubits 0, ..., k-1
        for q in range(n_qubits):
            if q < k:
                factors.append(_Z)
            elif q == k:
                factors.append(_Y if is_y else _X)
            else:
                factors.append(_I2)

        op = _kron_list(factors)
        ops.append(op)

    return ops


def _build_majorana_sparse(N, n_qubits):
    """Build sparse Majorana operators."""
    ops = []
    for i in range(N):
        k = i // 2
        is_y = i % 2

        factors = []
        for q in range(n_qubits):
            if q < k:
                factors.append(_Z_sp)
            elif q == k:
                factors.append(_Y_sp if is_y else _X_sp)
            else:
                factors.append(_I2_sp)

        op = _kron_list_sparse(factors)
        ops.append(op)

    return ops


def verify_anticommutation(ops, tol=1e-12):
    """Verify {psi_i, psi_j} = 2 delta_{ij} for all pairs.

    Returns
    -------
    max_error : float
        Maximum deviation from the expected anticommutation relation.
    """
    N = len(ops)
    max_error = 0.0
    is_sparse = sparse.issparse(ops[0])

    for i in range(N):
        for j in range(i, N):
            anticomm = ops[i] @ ops[j] + ops[j] @ ops[i]
            expected = 2.0 if i == j else 0.0

            if is_sparse:
                if i == j:
                    diff = anticomm - expected * sparse.eye(anticomm.shape[0], format='csr')
                    err = sparse.linalg.norm(diff)
                else:
                    err = sparse.linalg.norm(anticomm)
            else:
                if i == j:
                    diff = anticomm - expected * np.eye(anticomm.shape[0])
                    err = np.linalg.norm(diff)
                else:
                    err = np.linalg.norm(anticomm)

            max_error = max(max_error, err)

    return max_error


def verify_hermiticity(ops, tol=1e-12):
    """Verify all operators are Hermitian.

    Returns
    -------
    max_error : float
        Maximum deviation from Hermiticity across all operators.
    """
    max_error = 0.0
    is_sparse = sparse.issparse(ops[0])

    for op in ops:
        if is_sparse:
            diff = op - op.conj().T
            err = sparse.linalg.norm(diff)
        else:
            diff = op - op.conj().T
            err = np.linalg.norm(diff)
        max_error = max(max_error, err)

    return max_error
