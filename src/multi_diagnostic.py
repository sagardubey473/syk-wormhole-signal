"""Multi-diagnostic holographic classifier.

Combines individual diagnostics into a single scoring rule for
"holographic vs not." Each diagnostic is mapped to a score in [0, 1]
where 1 = "consistent with holographic SYK" and 0 = "non-holographic limit."

Diagnostics:
    s_H:      Transmission peak height (already in [0, 1])
    s_r:      Level spacing ratio (Poisson=0.39 → 0, GUE=0.60 → 1)
    s_lambda: Lyapunov / chaos bound (0 → 0, saturated → 1)
    s_SFF:    SFF ramp quality (no ramp → 0, linear ramp → 1)

Combined scores:
    S_geom = (s_H * s_r * s_lambda * s_SFF)^(1/4)
    S_min  = min(s_H, s_r, s_lambda, s_SFF)

Gap 1 investigation: does this classifier discriminate
holographic from non-holographic systems?
"""

import numpy as np


# Reference values for score normalization
R_POISSON = 0.3863   # Poisson level spacing ratio
R_GUE = 0.6027       # GUE level spacing ratio


def score_transmission(peak_height):
    """Score from transmission peak height.

    H = |C|_max already in [0, 1]. Clip for safety.

    Parameters
    ----------
    peak_height : float or ndarray

    Returns
    -------
    s_H : float or ndarray, in [0, 1]
    """
    return np.clip(peak_height, 0.0, 1.0)


def score_level_spacing(r_mean):
    """Score from level spacing ratio.

    Linear interpolation: Poisson (0.3863) → 0, GUE (0.6027) → 1.
    Clipped to [0, 1].

    Parameters
    ----------
    r_mean : float or ndarray

    Returns
    -------
    s_r : float or ndarray, in [0, 1]
    """
    s = (r_mean - R_POISSON) / (R_GUE - R_POISSON)
    return np.clip(s, 0.0, 1.0)


def score_lyapunov(lambda_L, beta):
    """Score from Lyapunov exponent relative to chaos bound.

    Chaos bound: lambda_max = 2*pi/beta.
    Score = lambda_L / lambda_max, clipped to [0, 1].

    Parameters
    ----------
    lambda_L : float or ndarray
        Extracted Lyapunov exponent.
    beta : float
        Inverse temperature.

    Returns
    -------
    s_lambda : float or ndarray, in [0, 1]
    """
    lambda_max = 2.0 * np.pi / beta
    s = lambda_L / lambda_max
    return np.clip(s, 0.0, 1.0)


def score_sff_ramp(K_values, t_values, dim):
    """Score from spectral form factor ramp quality.

    A "good" ramp means K(t) grows linearly from the dip minimum
    to the plateau. We measure the correlation between K(t) and t
    in the ramp region.

    The ramp region is identified as times where K(t) is between
    K_min (dip) and K_plateau ≈ 1/dim.

    Parameters
    ----------
    K_values : ndarray
        Spectral form factor values.
    t_values : ndarray
        Time points.
    dim : int
        Hilbert space dimension.

    Returns
    -------
    s_SFF : float, in [0, 1]
    """
    K_plateau = 1.0 / dim
    K_min = np.min(K_values)

    # Find ramp region: between dip minimum and 80% of plateau
    dip_idx = np.argmin(K_values)
    ramp_mask = (np.arange(len(K_values)) >= dip_idx) & \
                (K_values < 0.8 * K_plateau) & \
                (K_values > K_min)

    if np.sum(ramp_mask) < 3:
        # Not enough points to identify a ramp
        return 0.0

    t_ramp = t_values[ramp_mask]
    K_ramp = K_values[ramp_mask]

    # Pearson correlation between K and t in ramp region
    if np.std(t_ramp) == 0 or np.std(K_ramp) == 0:
        return 0.0

    corr = np.corrcoef(t_ramp, K_ramp)[0, 1]
    # Map correlation to [0, 1]: corr=1 means perfect linear ramp
    return np.clip(corr, 0.0, 1.0)


def combined_score_geometric(s_H, s_r, s_lambda, s_SFF):
    """Geometric mean of individual scores.

    S_geom = (s_H * s_r * s_lambda * s_SFF)^(1/4)

    All inputs clipped to [epsilon, 1] to avoid zero domination.
    """
    eps = 1e-3
    scores = np.array([
        np.clip(s_H, eps, 1.0),
        np.clip(s_r, eps, 1.0),
        np.clip(s_lambda, eps, 1.0),
        np.clip(s_SFF, eps, 1.0)
    ])
    return np.prod(scores) ** 0.25


def combined_score_minimum(s_H, s_r, s_lambda, s_SFF):
    """Minimum of individual scores.

    S_min = min(s_H, s_r, s_lambda, s_SFF)

    This is the most conservative classifier: requires ALL diagnostics
    to be holographic-like.
    """
    return min(s_H, s_r, s_lambda, s_SFF)


def classify_system(s_H, s_r, s_lambda, s_SFF):
    """Classify a system using both combined scores.

    Returns
    -------
    dict with keys:
        's_H', 's_r', 's_lambda', 's_SFF': individual scores
        'S_geom': geometric mean score
        'S_min': minimum score
        'classification': 'holographic', 'marginal', or 'non-holographic'
    """
    S_geom = combined_score_geometric(s_H, s_r, s_lambda, s_SFF)
    S_min = combined_score_minimum(s_H, s_r, s_lambda, s_SFF)

    # Classification thresholds (to be validated in Gap 1)
    if S_min >= 0.7:
        classification = 'holographic'
    elif S_min >= 0.3:
        classification = 'marginal'
    else:
        classification = 'non-holographic'

    return {
        's_H': s_H,
        's_r': s_r,
        's_lambda': s_lambda,
        's_SFF': s_SFF,
        'S_geom': S_geom,
        'S_min': S_min,
        'classification': classification
    }
