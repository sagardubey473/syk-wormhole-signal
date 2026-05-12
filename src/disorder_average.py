"""Disorder averaging utilities.

Correct averaging: average |C(t)| (not C(t) directly), since phases
differ across disorder realizations. Report standard error of the mean.
"""

import numpy as np
from joblib import Parallel, delayed
import os


def disorder_average(func, n_realizations, n_jobs=None, seeds=None, **kwargs):
    """Run a function over many disorder realizations and average.

    Parameters
    ----------
    func : callable
        Function that takes (seed, **kwargs) and returns a result array.
    n_realizations : int
        Number of disorder realizations.
    n_jobs : int or None
        Number of parallel jobs. None = sequential.
    seeds : list or None
        Explicit seeds. If None, use range(n_realizations).
    **kwargs
        Additional keyword arguments passed to func.

    Returns
    -------
    mean : ndarray
        Mean across realizations.
    stderr : ndarray
        Standard error of the mean.
    all_results : list of ndarray
        Individual results for each realization.
    """
    if seeds is None:
        seeds = list(range(n_realizations))

    if n_jobs is None or n_jobs == 1:
        results = [func(seed=s, **kwargs) for s in seeds]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(func)(seed=s, **kwargs) for s in seeds
        )

    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0, ddof=1)
    stderr = std / np.sqrt(len(results))

    return mean, stderr, results


def disorder_average_abs(func, n_realizations, n_jobs=None, seeds=None, **kwargs):
    """Same as disorder_average but takes |result| before averaging.

    Appropriate for transmission signals where phases differ across realizations.
    """
    if seeds is None:
        seeds = list(range(n_realizations))

    if n_jobs is None or n_jobs == 1:
        results = [np.abs(func(seed=s, **kwargs)) for s in seeds]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(lambda s: np.abs(func(seed=s, **kwargs)))(s) for s in seeds
        )

    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0, ddof=1)
    stderr = std / np.sqrt(len(results))

    return mean, stderr, results


def save_results(filepath, **data):
    """Save results to NPZ file.

    Parameters
    ----------
    filepath : str
        Path to save file.
    **data
        Named arrays to save.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **data)


def load_results(filepath):
    """Load results from NPZ file.

    Returns
    -------
    data : dict-like
        NpzFile object with named arrays.
    """
    return np.load(filepath, allow_pickle=True)


def check_cache(filepath):
    """Check if a cached result file exists.

    Returns
    -------
    exists : bool
    """
    return os.path.exists(filepath)
