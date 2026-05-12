"""Observable computation utilities.

Functions for computing physical observables used across research questions:
- Level spacing ratio statistics
- OTOC and Lyapunov exponent extraction
- Two-point functions
- Spectral form factor
- Mutual information
"""

import numpy as np
from scipy import sparse


def level_spacing_ratio(eigenvalues):
    """Compute level spacing ratios r_n = min(s_n, s_{n+1})/max(s_n, s_{n+1}).

    Parameters
    ----------
    eigenvalues : ndarray
        Sorted eigenvalues.

    Returns
    -------
    r_values : ndarray
    r_mean : float
    """
    evals = np.sort(eigenvalues)
    spacings = np.diff(evals)
    # Remove near-zero spacings (exact degeneracies)
    spacings = spacings[spacings > 1e-14]

    if len(spacings) < 2:
        return np.array([]), 0.0

    r_values = np.minimum(spacings[:-1], spacings[1:]) / \
               np.maximum(spacings[:-1], spacings[1:])
    return r_values, np.mean(r_values)


def spectral_form_factor(eigenvalues, t_array):
    """Compute spectral form factor K(t) = |Tr(e^{-iHt})|^2 / dim^2.

    The normalization by dim^2 gives K(t->inf) -> 1/dim for GUE.

    Parameters
    ----------
    eigenvalues : ndarray
    t_array : ndarray

    Returns
    -------
    K : ndarray
    """
    dim = len(eigenvalues)
    K = np.zeros(len(t_array))
    for idx, t in enumerate(t_array):
        z = np.sum(np.exp(-1j * eigenvalues * t))
        K[idx] = np.abs(z)**2 / dim**2
    return K


def extract_lyapunov(F, t_array, F0=None, fit_range=None):
    """Extract Lyapunov exponent from OTOC decay.

    The OTOC F(t) at early times satisfies:
    1 - F(t)/F(0) ~ epsilon * exp(lambda_L * t)

    We fit log(1 - F(t)/F(0)) to extract lambda_L.

    Parameters
    ----------
    F : ndarray, complex
        OTOC values.
    t_array : ndarray
        Time points.
    F0 : float or None
        F(t=0). If None, use F[0].
    fit_range : tuple (t_min, t_max) or None
        Time range for the linear fit. If None, auto-detect.

    Returns
    -------
    lambda_L : float
        Lyapunov exponent.
    fit_quality : float
        R^2 of the linear fit.
    fit_range_used : tuple
        (t_min, t_max) actually used.
    """
    F_real = np.real(F)
    if F0 is None:
        F0 = F_real[0]

    # Compute 1 - F/F0
    decay = 1.0 - F_real / F0

    # Find the region where decay is positive and growing exponentially
    positive = decay > 1e-10
    if not np.any(positive):
        return 0.0, 0.0, (0.0, 0.0)

    log_decay = np.full_like(decay, np.nan)
    log_decay[positive] = np.log(decay[positive])

    if fit_range is not None:
        t_min, t_max = fit_range
        mask = (t_array >= t_min) & (t_array <= t_max) & positive
    else:
        # Auto-detect: use the region where log_decay is approximately linear
        # Take the middle portion where decay is between 0.01 and 0.5
        mask = positive & (decay > 0.01) & (decay < 0.5)
        if np.sum(mask) < 3:
            # Fall back to any positive decay region
            mask = positive & np.isfinite(log_decay)

    if np.sum(mask) < 2:
        return 0.0, 0.0, (0.0, 0.0)

    t_fit = t_array[mask]
    y_fit = log_decay[mask]

    # Linear fit: y = lambda_L * t + const
    coeffs = np.polyfit(t_fit, y_fit, 1)
    lambda_L = coeffs[0]

    # R^2
    y_pred = np.polyval(coeffs, t_fit)
    ss_res = np.sum((y_fit - y_pred)**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return lambda_L, r_squared, (t_fit[0], t_fit[-1])


def mutual_information(state, dim_L, dim_R):
    """Compute mutual information I(L:R) = S_L + S_R - S_LR.

    For a pure state, S_LR = 0, so I = 2 * S_L = 2 * S_R.

    Parameters
    ----------
    state : ndarray
        State vector.
    dim_L, dim_R : int

    Returns
    -------
    I : float
    """
    from .tfd import entanglement_entropy
    S = entanglement_entropy(state, dim_L, dim_R)
    # For pure state, I(L:R) = 2*S
    return 2.0 * S


def thermal_two_point_vectorized(evals, evecs, psi_op, beta, t_array):
    """Compute thermal two-point function G(t) vectorized over time.

    G(t) = Tr[rho * psi(t) * psi(0)]
         = sum_{m,n} rho_m * |<m|psi|n>|^2 * e^{i(E_m-E_n)t}

    Parameters
    ----------
    evals : ndarray, shape (dim,)
    evecs : ndarray, shape (dim, dim)
    psi_op : ndarray, shape (dim, dim)
    beta : float
    t_array : ndarray

    Returns
    -------
    G : ndarray, shape (len(t_array),), complex
    """
    boltzmann = np.exp(-beta * evals)
    Z = np.sum(boltzmann)
    rho_m = boltzmann / Z

    # psi in eigenbasis
    if sparse.issparse(psi_op):
        psi_op = psi_op.toarray()
    psi_eig = evecs.conj().T @ psi_op @ evecs

    # A[m,n] = rho_m * |psi_{mn}|^2
    # But actually G(t) = sum_{m,n} rho_m psi_{mn} psi_{nm} e^{i(E_m-E_n)t}
    A = rho_m[:, None] * psi_eig * psi_eig.T.conj()

    # Energy differences
    dE = np.subtract.outer(evals, evals)

    G = np.zeros(len(t_array), dtype=complex)
    for idx, t in enumerate(t_array):
        G[idx] = np.sum(A * np.exp(1j * dE * t))

    return G
