"""Wormhole transmission signal computation.

Computes C(t) = <TFD| psi^R_j(t) psi^L_j(0) |TFD> where psi^R_j(t) is
evolved under the full coupled Hamiltonian H = H_L + H_R + H_int.

See CONVENTIONS.md section 9.

References:
    - Maldacena & Qi, arXiv:1804.00491, eq. (4.1)-(4.5), Figure 2
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


def transmission_signal(H_coupled, tfd, psi_L, psi_R, t_array,
                        sites=None, use_eigen=True):
    """Compute wormhole transmission signal C(t).

    C_j(t) = <TFD| psi^R_j(t) psi^L_j(0) |TFD>
           = <TFD| e^{iHt} psi^R_j e^{-iHt} psi^L_j |TFD>

    Averaged over sites j.

    Parameters
    ----------
    H_coupled : ndarray or sparse, shape (dim, dim)
        Full coupled Hamiltonian H = H_L + H_R + H_int.
    tfd : ndarray, shape (dim,)
        TFD state vector.
    psi_L : list of ndarray/sparse
        Left Majorana operators in the doubled space.
    psi_R : list of ndarray/sparse
        Right Majorana operators in the doubled space.
    t_array : ndarray
        Time points.
    sites : list of int or None
        Majorana site indices to average over. If None, use all.
    use_eigen : bool
        If True, use eigendecomposition for time evolution (exact, good
        for moderate dim). If False, use Krylov methods (for large dim).

    Returns
    -------
    C : ndarray, shape (len(t_array),), complex
        Site-averaged transmission signal.
    """
    N_per_side = len(psi_L)
    if sites is None:
        sites = list(range(N_per_side))

    dim = len(tfd)

    if use_eigen:
        return _transmission_eigen(H_coupled, tfd, psi_L, psi_R, t_array, sites)
    else:
        return _transmission_krylov(H_coupled, tfd, psi_L, psi_R, t_array, sites)


def _transmission_eigen(H_coupled, tfd, psi_L, psi_R, t_array, sites):
    """Eigendecomposition-based time evolution.

    C_j(t) = <TFD| e^{iHt} psi^R_j e^{-iHt} psi^L_j |TFD>

    Compute by:
    1. |phi_j> = psi^L_j |TFD>
    2. |chi_j> = psi^R_j |TFD>  (note: <TFD|psi^R_j = <chi_j| since psi^R is Hermitian)
    3. C_j(t) = <chi_j| e^{iHt} psi^R_j e^{-iHt} |phi_j>
       Wait, that's wrong. Let me re-derive:

    C_j(t) = <TFD| e^{iHt} psi^R_j e^{-iHt} psi^L_j |TFD>
    Let |phi_j> = psi^L_j |TFD>
    Let |alpha(t)> = e^{-iHt} |phi_j>
    Then C_j(t) = <TFD| e^{iHt} psi^R_j |alpha(t)>
    Let |beta(t)> = psi^R_j |alpha(t)>
    Then C_j(t) = <TFD| e^{iHt} |beta(t)>

    But this requires computing e^{iHt} at each t, which is expensive.

    Better approach using eigenbasis:
    Diagonalize H = V diag(E) V^dagger.
    e^{-iHt} = V diag(e^{-iEt}) V^dagger.

    |phi_j> = psi^L_j |TFD>
    |gamma_j> = V^dagger psi^R_j^dagger (= psi^R_j since Hermitian)
    Wait, let me just use the standard approach.

    C_j(t) = <bra| e^{iHt} psi^R_j e^{-iHt} |ket>
    where |bra> = |TFD>, |ket> = psi^L_j |TFD>.

    In eigenbasis:
    = sum_{m,n} <bra|m> e^{i E_m t} <m|psi^R_j|n> e^{-i E_n t} <n|ket>
    = sum_{m,n} bra_m^* R_{mn} ket_n e^{i(E_m - E_n)t}

    where bra_m = <m|TFD>, R_{mn} = <m|psi^R_j|n>, ket_n = <n|psi^L_j|TFD>.
    """
    if sparse.issparse(H_coupled):
        H_dense = H_coupled.toarray()
    else:
        H_dense = H_coupled

    # Ensure exact Hermiticity
    H_herm = 0.5 * (H_dense + H_dense.conj().T)
    evals, evecs = np.linalg.eigh(H_herm)

    dim = len(tfd)
    # TFD in eigenbasis
    tfd_eig = evecs.conj().T @ tfd  # bra_m = <m|TFD>

    C = np.zeros(len(t_array), dtype=complex)

    for j in sites:
        # psi^L_j and psi^R_j in the computational basis
        pL = psi_L[j]
        pR = psi_R[j]
        if sparse.issparse(pL):
            pL = pL.toarray()
        if sparse.issparse(pR):
            pR = pR.toarray()

        # |ket> = psi^L_j |TFD>
        ket = pL @ tfd
        ket_eig = evecs.conj().T @ ket  # <n|ket>

        # psi^R_j in eigenbasis
        R_eig = evecs.conj().T @ pR @ evecs  # <m|psi^R_j|n>

        # C_j(t) = sum_{m,n} tfd_eig[m]^* R_eig[m,n] ket_eig[n] e^{i(E_m-E_n)t}
        # = sum_{m,n} A[m,n] e^{i(E_m-E_n)t}
        # where A[m,n] = conj(tfd_eig[m]) * R_eig[m,n] * ket_eig[n]

        A = np.outer(tfd_eig.conj(), ket_eig) * R_eig  # shape (dim, dim)

        for idx, t in enumerate(t_array):
            phases = np.exp(1j * np.subtract.outer(evals, evals) * t)  # e^{i(E_m-E_n)t}
            C[idx] += np.sum(A * phases)

    C /= len(sites)
    return C


def _transmission_eigen_fast(H_coupled, tfd, psi_L, psi_R, t_array, sites):
    """Faster eigendecomposition method using vectorized phase computation."""
    if sparse.issparse(H_coupled):
        H_dense = H_coupled.toarray()
    else:
        H_dense = H_coupled

    H_herm = 0.5 * (H_dense + H_dense.conj().T)
    evals, evecs = np.linalg.eigh(H_herm)

    tfd_eig = evecs.conj().T @ tfd

    # Precompute energy differences
    dE = np.subtract.outer(evals, evals)  # (dim, dim)

    C = np.zeros(len(t_array), dtype=complex)

    for j in sites:
        pL = psi_L[j].toarray() if sparse.issparse(psi_L[j]) else psi_L[j]
        pR = psi_R[j].toarray() if sparse.issparse(psi_R[j]) else psi_R[j]

        ket = pL @ tfd
        ket_eig = evecs.conj().T @ ket

        R_eig = evecs.conj().T @ pR @ evecs

        A = np.outer(tfd_eig.conj(), ket_eig) * R_eig

        # Vectorized over time
        for idx, t in enumerate(t_array):
            C[idx] += np.sum(A * np.exp(1j * dE * t))

    C /= len(sites)
    return C


def _transmission_krylov(H_coupled, tfd, psi_L, psi_R, t_array, sites):
    """Krylov-based time evolution for large systems.

    Uses scipy.sparse.linalg.expm_multiply for e^{-iHt}|v>.
    """
    if not sparse.issparse(H_coupled):
        H_coupled = sparse.csr_matrix(H_coupled)

    C = np.zeros(len(t_array), dtype=complex)

    for j in sites:
        pL = psi_L[j]
        pR = psi_R[j]

        # |ket> = psi^L_j |TFD>
        ket = pL @ tfd

        for idx, t in enumerate(t_array):
            # e^{-iHt} |ket>
            evolved = expm_multiply(-1j * H_coupled * t, ket)
            # psi^R_j |evolved>
            acted = pR @ evolved
            # <TFD| e^{iHt} |acted> = <e^{-iHt} TFD| acted>
            tfd_evolved = expm_multiply(-1j * H_coupled * t, tfd)
            C[idx] += tfd_evolved.conj() @ acted

    C /= len(sites)
    return C


def extract_peak(C, t_array):
    """Extract peak height, time, and FWHM from transmission signal.

    FWHM is computed by searching left and right from the peak for the
    first crossing below half-maximum, with linear interpolation for
    sub-grid accuracy. Returns NaN if the signal does not drop below
    half-max on either side of the peak.

    Parameters
    ----------
    C : ndarray, complex
        Transmission signal.
    t_array : ndarray
        Time points.

    Returns
    -------
    peak_height : float
        Maximum |C(t)|.
    peak_time : float
        Time at which maximum occurs.
    fwhm : float
        Full width at half maximum. NaN if unresolved on either side.
    """
    absC = np.abs(C)
    peak_idx = np.argmax(absC)
    peak_height = absC[peak_idx]
    peak_time = t_array[peak_idx]

    half_max = peak_height / 2.0

    # Search LEFT from peak for first crossing below half_max
    t_left = np.nan
    for i in range(peak_idx, 0, -1):
        if absC[i - 1] < half_max:
            # Linear interpolation between i-1 (below) and i (above)
            f_lo = absC[i - 1]
            f_hi = absC[i]
            frac = (half_max - f_lo) / (f_hi - f_lo)
            t_left = t_array[i - 1] + frac * (t_array[i] - t_array[i - 1])
            break

    # Search RIGHT from peak for first crossing below half_max
    t_right = np.nan
    for i in range(peak_idx, len(absC) - 1):
        if absC[i + 1] < half_max:
            # Linear interpolation between i (above) and i+1 (below)
            f_hi = absC[i]
            f_lo = absC[i + 1]
            frac = (half_max - f_hi) / (f_lo - f_hi)
            t_right = t_array[i] + frac * (t_array[i + 1] - t_array[i])
            break

    if np.isnan(t_left) or np.isnan(t_right):
        fwhm = np.nan
    else:
        fwhm = t_right - t_left

    return peak_height, peak_time, fwhm
