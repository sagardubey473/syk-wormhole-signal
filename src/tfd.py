"""Thermofield double (TFD) state construction.

|TFD(beta)> = (1/sqrt(Z)) sum_n exp(-beta*E_n/2) |n>_L |n>_R

In the doubled Hilbert space with 2*N_per_side Majorana fermions.

In our Jordan-Wigner construction, H_L and H_R (restricted to their
respective tensor factors) are represented by the SAME matrix as
H_single. Therefore the TFD uses |n>_R (NOT |n*>_R) — both sides
use the same eigenvectors without conjugation.

This differs from the abstract MQ convention |TFD> = Σ e^{-βE/2} |n>|n*>
where the conjugation is needed because the right Hamiltonian is defined
as H_R = K H_L K^{-1} with K the anti-unitary time-reversal operator.
In our explicit qubit representation, the right Majoranas have the same
matrix structure as the left, so no conjugation is needed.

See CONVENTIONS.md section 8.

References:
    - Maldacena & Qi, arXiv:1804.00491, eq. (2.4)
"""

import numpy as np
from scipy import sparse


def build_tfd(doubled_syk, beta):
    """Construct the thermofield double state in the doubled Hilbert space.

    The TFD is constructed by:
    1. Diagonalizing H_L in the doubled space to get |n>_L eigenvectors.
    2. These eigenvectors are in the full doubled Hilbert space basis.
    3. Since H_L acts trivially on the right side, eigenstates of H_L factor
       as |E_m>_L x |anything>_R. We need to carefully construct the TFD
       using the tensor product structure.

    For the TFD, we use:
        |TFD> = (1/sqrt(Z)) sum_n exp(-beta*E_n/2) |n>_L x |n>_R

    where |n> are eigenstates of the single-copy SYK Hamiltonian.
    No conjugation is applied to the right eigenvectors because H_R
    restricted to the right tensor factor is the same matrix as H_single
    (same JW structure, same couplings).

    Implementation: Build TFD in computational basis using the single-copy
    eigenstates and the known tensor product structure.

    Parameters
    ----------
    doubled_syk : DoubledSYK
        The doubled system.
    beta : float
        Inverse temperature.

    Returns
    -------
    tfd : ndarray, shape (dim,)
        TFD state vector in the doubled Hilbert space.
    Z : float
        Partition function.
    """
    N = doubled_syk.N_per_side
    dim_single = 2 ** (N // 2)  # dimension of single-side Hilbert space
    dim_total = doubled_syk.dim  # = dim_single^2

    # Get single-side SYK Hamiltonian by building one from the same couplings
    # Extract H_L restricted to the left factor of the tensor product
    # Since H_L = H_single x I_R, we need the single-side eigenstates.

    # Build single-side Hamiltonian using the same Majorana operators as
    # the standalone SYK (not the doubled ones)
    from .syk import SYKHamiltonian
    syk_single = SYKHamiltonian(N, seed=doubled_syk.seed, J=doubled_syk.J,
                                 sparsity=doubled_syk.sparsity, use_sparse=False)
    evals, evecs = syk_single.diagonalize()

    # evecs: shape (dim_single, dim_single), column n is |n>
    # evals: shape (dim_single,)

    # Partition function
    boltzmann = np.exp(-beta * evals)
    Z = np.sum(boltzmann)

    # Build TFD state
    # |TFD> = (1/sqrt(Z)) sum_n exp(-beta*E_n/2) |n>_L x |n>_R
    # No conjugation on right side since H_R_restricted = H_single (same matrix).

    # In computational basis, |n>_L = sum_a evecs[a,n] |a>_L
    # |n>_R = sum_b evecs[b,n] |b>_R
    # |TFD> = (1/sqrt(Z)) sum_n exp(-beta*E_n/2) sum_{a,b} evecs[a,n] evecs[b,n] |a>_L|b>_R

    # The doubled Hilbert space basis is |a>_L|b>_R with index a*dim_single + b
    # (left is the slower index, matching standard Kronecker product)

    tfd = np.zeros(dim_total, dtype=complex)
    weights = np.exp(-beta * evals / 2.0) / np.sqrt(Z)

    for n in range(dim_single):
        # |n>_L x |n>_R contribution
        # In Kronecker product convention: |a,b> -> index a*dim_single + b
        v_L = evecs[:, n]    # |n>_L coefficients
        v_R = evecs[:, n]    # |n>_R coefficients (same, no conjugation)

        # Outer product gives the contribution to the state vector
        # tfd[a*dim_single + b] += weight * v_L[a] * v_R[b]
        contrib = weights[n] * np.outer(v_L, v_R).ravel()
        tfd += contrib

    return tfd, Z


def partial_trace_R(state, dim_L, dim_R):
    """Compute the partial trace over the right subsystem.

    rho_L = Tr_R(|state><state|)

    Parameters
    ----------
    state : ndarray, shape (dim_L * dim_R,)
    dim_L : int
    dim_R : int

    Returns
    -------
    rho_L : ndarray, shape (dim_L, dim_L)
    """
    # Reshape state as matrix: psi[a, b] where a is left, b is right
    psi = state.reshape(dim_L, dim_R)
    # rho_L = psi @ psi^dagger
    rho_L = psi @ psi.conj().T
    return rho_L


def partial_trace_L(state, dim_L, dim_R):
    """Compute the partial trace over the left subsystem.

    rho_R = Tr_L(|state><state|)

    Parameters
    ----------
    state : ndarray, shape (dim_L * dim_R,)
    dim_L : int
    dim_R : int

    Returns
    -------
    rho_R : ndarray, shape (dim_R, dim_R)
    """
    psi = state.reshape(dim_L, dim_R)
    rho_R = psi.conj().T @ psi
    return rho_R


def entanglement_entropy(state, dim_L, dim_R):
    """Compute entanglement entropy S = -Tr(rho_L log rho_L).

    Parameters
    ----------
    state : ndarray
    dim_L, dim_R : int

    Returns
    -------
    S : float
    """
    rho_L = partial_trace_R(state, dim_L, dim_R)
    eigenvalues = np.linalg.eigvalsh(rho_L)
    # Filter out zero/negative eigenvalues for log
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    S = -np.sum(eigenvalues * np.log(eigenvalues))
    return S
