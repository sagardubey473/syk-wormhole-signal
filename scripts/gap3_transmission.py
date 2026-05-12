"""Gap 3: Transmission signal computation at proper statistics.

N=10 (primary test), 50 realizations, 9 sparsity values.
Uses multiprocessing for parallelism.

N=14 transmission is computationally prohibitive (dim=16384, ~100x slower)
so we only compute level spacing for N=14 (see gap3_level_spacing.py).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import time

from src.doubled import DoubledSYK
from src.tfd import build_tfd
from src.wormhole import extract_peak
from scipy import sparse


# Parameters
N_PER_SIDE = 10
BETA = 8.0
MU = 0.1
SPARSITY_VALUES = [1.0, 0.5, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.02]
N_REALIZATIONS = 50
T_MAX = 30.0
N_T = 120  # time points (sufficient to resolve peak at t~7)

t_array = np.linspace(0, T_MAX, N_T)


def transmission_signal_optimized(H_coupled, tfd, psi_L, psi_R, t_array):
    """Optimized transmission signal: precompute phases outside site loop.

    Same physics as src/wormhole.py:_transmission_eigen but avoids
    redundant exp computations (phases only depend on eigenvalues, not site).
    """
    if sparse.issparse(H_coupled):
        H_dense = H_coupled.toarray()
    else:
        H_dense = H_coupled

    H_herm = 0.5 * (H_dense + H_dense.conj().T)
    evals, evecs = np.linalg.eigh(H_herm)

    N_sites = len(psi_L)
    tfd_eig = evecs.conj().T @ tfd

    # Precompute energy differences
    dE = np.subtract.outer(evals, evals)  # (dim, dim)

    # Precompute A matrices for each site
    A_list = []
    for j in range(N_sites):
        pL = psi_L[j].toarray() if sparse.issparse(psi_L[j]) else psi_L[j]
        pR = psi_R[j].toarray() if sparse.issparse(psi_R[j]) else psi_R[j]

        ket = pL @ tfd
        ket_eig = evecs.conj().T @ ket
        R_eig = evecs.conj().T @ pR @ evecs
        A_list.append(np.outer(tfd_eig.conj(), ket_eig) * R_eig)

    # Sum A matrices (since we average over sites anyway)
    A_total = sum(A_list) / N_sites

    # Compute signal: one exp per time point (not per site)
    C = np.zeros(len(t_array), dtype=complex)
    for idx, t in enumerate(t_array):
        phases = np.exp(1j * dE * t)
        C[idx] = np.sum(A_total * phases)

    return C


def compute_one_realization(args):
    """Compute transmission peak for one (sparsity, seed) pair."""
    p, seed = args
    try:
        # Build doubled system (use_sparse=None lets it auto-select sparse
        # for construction, which is much faster for dim=1024)
        doubled = DoubledSYK(N_PER_SIDE, seed=seed, sparsity=p, use_sparse=None)

        # Build coupled Hamiltonian
        H_coupled = doubled.build_coupled_hamiltonian(MU)

        # Build TFD
        tfd, Z = build_tfd(doubled, BETA)

        # Compute transmission signal (optimized version)
        C = transmission_signal_optimized(H_coupled, tfd,
                                          doubled.psi_L, doubled.psi_R, t_array)

        # Extract peak
        peak_height, peak_time, fwhm = extract_peak(C, t_array)

        return {
            'seed': seed,
            'sparsity': p,
            'peak_height': peak_height,
            'peak_time': peak_time,
            'fwhm': fwhm,
            'C_abs_max': np.max(np.abs(C)),
            'success': True
        }
    except Exception as e:
        return {
            'seed': seed,
            'sparsity': p,
            'peak_height': np.nan,
            'peak_time': np.nan,
            'fwhm': np.nan,
            'C_abs_max': np.nan,
            'success': False,
            'error': str(e)
        }


if __name__ == '__main__':
    print(f"Gap 3: Transmission signal computation")
    print(f"  N_per_side = {N_PER_SIDE} (doubled dim = {2**N_PER_SIDE})")
    print(f"  beta = {BETA}, mu = {MU}")
    print(f"  Sparsity values: {SPARSITY_VALUES}")
    print(f"  Realizations: {N_REALIZATIONS}")
    print(f"  Time: 0 to {T_MAX}, {N_T} points")
    total = len(SPARSITY_VALUES) * N_REALIZATIONS
    print(f"  Total instances: {total}")
    print(f"  Running serially (~3s/instance, est. ~{total*3//60} min total)")
    print()

    # Build task list
    tasks = []
    for p in SPARSITY_VALUES:
        for seed in range(N_REALIZATIONS):
            tasks.append((p, seed))

    t_start = time.time()

    # Run serially (multiprocessing has startup issues on this platform)
    all_results = []
    for i, task in enumerate(tasks):
        result = compute_one_realization(task)
        all_results.append(result)
        if (i + 1) % 25 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            p_curr = task[0]
            print(f"  [{i+1}/{total}] p={p_curr:.3f} {elapsed:.0f}s elapsed, "
                  f"ETA {eta:.0f}s, rate={rate:.2f}/s", flush=True)

    total_time = time.time() - t_start
    print(f"\nAll {total} instances complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Organize results by sparsity
    peaks_by_sparsity = {p: [] for p in SPARSITY_VALUES}
    times_by_sparsity = {p: [] for p in SPARSITY_VALUES}
    fwhm_by_sparsity = {p: [] for p in SPARSITY_VALUES}

    n_failures = 0
    for r in all_results:
        if r['success']:
            peaks_by_sparsity[r['sparsity']].append(r['peak_height'])
            times_by_sparsity[r['sparsity']].append(r['peak_time'])
            fwhm_by_sparsity[r['sparsity']].append(r['fwhm'])
        else:
            n_failures += 1
            print(f"  FAILURE: seed={r['seed']}, p={r['sparsity']}: {r.get('error','?')}")

    if n_failures > 0:
        print(f"\n  WARNING: {n_failures} failures out of {total} instances")

    # Save to HDF5
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'gap3_transmission.h5')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['N_per_side'] = N_PER_SIDE
        f.attrs['beta'] = BETA
        f.attrs['mu'] = MU
        f.attrs['sparsity_values'] = SPARSITY_VALUES
        f.attrs['n_realizations'] = N_REALIZATIONS
        f.attrs['t_max'] = T_MAX
        f.attrs['n_t'] = N_T
        f.attrs['seeds'] = list(range(N_REALIZATIONS))
        f.attrs['description'] = ('Gap 3: Transmission peaks at 50 realizations, '
                                  'N=10, beta=8, mu=0.1')
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['total_compute_time_s'] = total_time

        f.create_dataset('t_array', data=t_array)

        for p in SPARSITY_VALUES:
            grp = f.create_group(f'p_{p:.3f}')
            grp.create_dataset('peak_heights', data=np.array(peaks_by_sparsity[p]))
            grp.create_dataset('peak_times', data=np.array(times_by_sparsity[p]))
            grp.create_dataset('fwhm', data=np.array(fwhm_by_sparsity[p]))

    print(f"\nSaved to {output_path}")

    # Print summary table
    print("\n=== Summary: Peak Height vs Sparsity (N=10, beta=8, mu=0.1) ===")
    print(f"{'p':>6} {'mean':>8} {'SEM':>8} {'std':>8} {'min':>8} {'max':>8} {'n':>4}")
    print("-" * 55)

    dense_mean = None
    for p in SPARSITY_VALUES:
        peaks = np.array(peaks_by_sparsity[p])
        if len(peaks) == 0:
            print(f"{p:>6.3f}  NO DATA")
            continue
        mean = np.mean(peaks)
        std = np.std(peaks)
        sem = std / np.sqrt(len(peaks))
        if p == 1.0:
            dense_mean = mean
        print(f"{p:>6.3f} {mean:>8.4f} {sem:>8.4f} {std:>8.4f} "
              f"{np.min(peaks):>8.4f} {np.max(peaks):>8.4f} {len(peaks):>4}")

    if dense_mean is not None:
        print(f"\n=== Ratio to Dense (p=1.0, mean={dense_mean:.4f}) ===")
        print(f"{'p':>6} {'ratio':>8} {'|1-ratio|':>10} {'sig?':>6}")
        print("-" * 35)
        for p in SPARSITY_VALUES:
            peaks = np.array(peaks_by_sparsity[p])
            if len(peaks) == 0:
                continue
            mean = np.mean(peaks)
            sem = np.std(peaks) / np.sqrt(len(peaks))
            ratio = mean / dense_mean
            # Is the difference statistically significant?
            # Under null: means are equal. Combined SEM for difference:
            dense_peaks = np.array(peaks_by_sparsity[1.0])
            dense_sem = np.std(dense_peaks) / np.sqrt(len(dense_peaks))
            combined_sem = np.sqrt(sem**2 + dense_sem**2)
            z_score = abs(mean - dense_mean) / combined_sem if combined_sem > 0 else 0
            sig = "YES" if z_score > 2 else "no"
            print(f"{p:>6.3f} {ratio:>8.4f} {abs(1-ratio):>10.4f} {sig:>6} (z={z_score:.1f})")

    print("\n=== Peak Time vs Sparsity ===")
    print(f"{'p':>6} {'mean t*':>8} {'SEM':>8}")
    print("-" * 25)
    for p in SPARSITY_VALUES:
        times = np.array(times_by_sparsity[p])
        if len(times) == 0:
            continue
        print(f"{p:>6.3f} {np.mean(times):>8.2f} {np.std(times)/np.sqrt(len(times)):>8.2f}")
