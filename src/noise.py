"""Noise models for open-system evolution.

Lindblad master equation:
    drho/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})

Noise channels:
    - Single-qubit dephasing: L_k = sqrt(gamma_phi) Z_k
    - Single-qubit amplitude damping: L_k = sqrt(gamma_amp) sigma^-_k
    - Correlated noise: L = sqrt(gamma_corr) sum_k Z_k
    - Asymmetric noise: different rates on L vs R sides

References:
    - RQ3 investigation of noise robustness
"""

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp


def build_dephasing_ops(n_qubits, gamma_phi):
    """Build single-qubit dephasing Lindblad operators.

    L_k = sqrt(gamma_phi) * Z_k for each qubit k.

    Parameters
    ----------
    n_qubits : int
    gamma_phi : float

    Returns
    -------
    L_ops : list of ndarray
    """
    dim = 2 ** n_qubits
    L_ops = []
    sqrt_gamma = np.sqrt(gamma_phi)

    for k in range(n_qubits):
        # Z_k = I x ... x Z x ... x I
        op = np.eye(1)
        for q in range(n_qubits):
            if q == k:
                op = np.kron(op, np.array([[1, 0], [0, -1]], dtype=complex))
            else:
                op = np.kron(op, np.eye(2, dtype=complex))
        L_ops.append(sqrt_gamma * op)

    return L_ops


def build_amplitude_damping_ops(n_qubits, gamma_amp):
    """Build single-qubit amplitude damping Lindblad operators.

    L_k = sqrt(gamma_amp) * sigma^-_k for each qubit k.

    Parameters
    ----------
    n_qubits : int
    gamma_amp : float

    Returns
    -------
    L_ops : list of ndarray
    """
    dim = 2 ** n_qubits
    L_ops = []
    sqrt_gamma = np.sqrt(gamma_amp)
    sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)

    for k in range(n_qubits):
        op = np.eye(1)
        for q in range(n_qubits):
            if q == k:
                op = np.kron(op, sigma_minus)
            else:
                op = np.kron(op, np.eye(2, dtype=complex))
        L_ops.append(sqrt_gamma * op)

    return L_ops


def build_correlated_dephasing_ops(n_qubits, gamma_corr):
    """Build collective dephasing Lindblad operator.

    L = sqrt(gamma_corr) * sum_k Z_k

    Parameters
    ----------
    n_qubits : int
    gamma_corr : float

    Returns
    -------
    L_ops : list of ndarray (single element)
    """
    dim = 2 ** n_qubits
    sqrt_gamma = np.sqrt(gamma_corr)

    total_Z = np.zeros((dim, dim), dtype=complex)
    for k in range(n_qubits):
        op = np.eye(1)
        for q in range(n_qubits):
            if q == k:
                op = np.kron(op, np.array([[1, 0], [0, -1]], dtype=complex))
            else:
                op = np.kron(op, np.eye(2, dtype=complex))
        total_Z += op

    return [sqrt_gamma * total_Z]


def lindblad_rhs(t, rho_vec, H, L_ops, dim):
    """Right-hand side of the Lindblad master equation in vectorized form.

    Parameters
    ----------
    t : float
    rho_vec : ndarray, shape (dim^2,)
        Density matrix flattened in row-major order.
    H : ndarray, shape (dim, dim)
    L_ops : list of ndarray
    dim : int

    Returns
    -------
    drho_vec : ndarray, shape (dim^2,)
    """
    rho = rho_vec.reshape(dim, dim)

    # Hamiltonian part: -i[H, rho]
    drho = -1j * (H @ rho - rho @ H)

    # Dissipative part
    for L in L_ops:
        Ldag = L.conj().T
        LdL = Ldag @ L
        drho += L @ rho @ Ldag - 0.5 * (LdL @ rho + rho @ LdL)

    return drho.ravel()


def evolve_lindblad(rho0, H, L_ops, t_array, method='RK45'):
    """Evolve density matrix under Lindblad equation.

    Parameters
    ----------
    rho0 : ndarray, shape (dim, dim)
        Initial density matrix.
    H : ndarray, shape (dim, dim)
        Hamiltonian.
    L_ops : list of ndarray
        Lindblad operators.
    t_array : ndarray
        Time points.
    method : str
        ODE solver method.

    Returns
    -------
    rho_t : ndarray, shape (len(t_array), dim, dim)
        Density matrix at each time point.
    """
    dim = rho0.shape[0]
    rho0_vec = rho0.ravel()

    sol = solve_ivp(
        lindblad_rhs,
        (t_array[0], t_array[-1]),
        rho0_vec,
        t_eval=t_array,
        method=method,
        args=(H, L_ops, dim),
        rtol=1e-8,
        atol=1e-10
    )

    rho_t = sol.y.T.reshape(len(t_array), dim, dim)
    return rho_t


def noisy_transmission_signal(H, rho0, psi_L, psi_R, L_ops, t_array, sites=None):
    """Compute transmission signal under Lindblad noise.

    C(t) = Tr[rho(t) psi^R_j psi^L_j] where rho(t) evolves under Lindblad.

    Note: this is NOT the same as the pure-state correlator; it uses the
    Schrodinger picture with the density matrix evolving.

    For the wormhole protocol, rho0 = |TFD><TFD| with Majorana insertion:
    rho0 = psi^L_j |TFD><TFD| (appropriately symmetrized).

    Parameters
    ----------
    H : ndarray
    rho0 : ndarray
        Initial density matrix (after Majorana insertion).
    psi_L : list of ndarray
    psi_R : list of ndarray
    L_ops : list of ndarray
    t_array : ndarray
    sites : list of int or None

    Returns
    -------
    C : ndarray, complex
    """
    N_per_side = len(psi_L)
    if sites is None:
        sites = list(range(N_per_side))

    dim = H.shape[0]
    C = np.zeros(len(t_array), dtype=complex)

    for j in sites:
        pL = psi_L[j] if not sparse.issparse(psi_L[j]) else psi_L[j].toarray()
        pR = psi_R[j] if not sparse.issparse(psi_R[j]) else psi_R[j].toarray()

        # Initial state: psi^L_j |TFD><TFD|
        # Actually, for the correct protocol:
        # C(t) = Tr(psi^R_j(t) psi^L_j rho_TFD)
        # = Tr(psi^R_j rho_eff(t)) where rho_eff evolves from psi^L_j rho_TFD
        rho_init = pL @ rho0

        # Evolve rho_init under Lindblad
        rho_t = evolve_lindblad(rho_init, H, L_ops, t_array)

        for idx in range(len(t_array)):
            C[idx] += np.trace(pR @ rho_t[idx])

    C /= len(sites)
    return C
