"""Fix 3: N=14 transmission signal using Krylov time evolution.

Full eigendecomposition of the 16384x16384 coupled Hamiltonian exceeds the
5-minute threshold, so we use scipy.sparse.linalg.expm_multiply for time
evolution instead. This computes exp(iHt)|v> directly using Krylov methods
with the sparse Hamiltonian (nnz ~1.9M, density ~0.7%).

Scope (reduced per spec):
- N_per_side = 14 (dim = 16384)
- 3 sparsity values: p = {1.0, 0.3, 0.05}
- 30 realizations per sparsity
- beta = 8, mu = 0.1

Signal: C(t) = (1/N) sum_j <TFD| psi^R_j exp(iHt) psi^L_j |TFD>
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import time
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from src.doubled import DoubledSYK
from src.tfd import build_tfd
from src.wormhole import extract_peak

# Parameters
N_PER_SIDE = 14
BETA = 8.0
MU = 0.1
SPARSITY_VALUES = [1.0, 0.3, 0.05]
N_REALIZATIONS = 30
T_MAX = 30.0
N_T = 120

t_array = np.linspace(0, T_MAX, N_T)


def transmission_krylov(H_coupled, tfd, psi_L, psi_R, t_array):
    """Compute transmission signal using Krylov time evolution.

    Avoids full eigendecomposition by using expm_multiply to compute
    exp(iHt)|v> directly with the sparse Hamiltonian.

    C(t) = (1/N) sum_j <TFD| psi^R_j exp(iHt) psi^L_j |TFD>
    """
    N_sites = len(psi_L)
    n_t = len(t_array)

    # Ensure H is sparse CSC for efficient matvec
    if sparse.issparse(H_coupled):
        H_sp = H_coupled.tocsc()
    else:
        H_sp = sparse.csc_matrix(H_coupled)

    # The generator for time evolution: A = i*H
    # expm_multiply computes expm(t*A) @ v for the specified t values
    A = 1j * H_sp

    C_total = np.zeros(n_t, dtype=complex)

    for j in range(N_sites):
        # |v_j> = psi^L_j |TFD>
        pL_j = psi_L[j]
        if sparse.issparse(pL_j):
            v_j = pL_j @ tfd
        else:
            v_j = pL_j @ tfd

        # |w_j> = psi^R_j |TFD> (psi^R is Hermitian)
        pR_j = psi_R[j]
        if sparse.issparse(pR_j):
            w_j = pR_j @ tfd
        else:
            w_j = pR_j @ tfd

        # Compute exp(iHt)|v_j> for all t using Krylov subspace method
        # expm_multiply with start/stop/num computes at uniformly spaced points
        if t_array[0] == 0:
            # expm_multiply with start=0 includes the initial point (t=0)
            v_evolved = expm_multiply(A, v_j, start=0, stop=T_MAX,
                                       num=n_t, endpoint=True)
            # v_evolved is (n_t, dim) array: v_evolved[k] = exp(iH*t_k)|v_j>
        else:
            v_evolved = expm_multiply(A, v_j, start=t_array[0], stop=t_array[-1],
                                       num=n_t, endpoint=True)

        # C_j(t) = <w_j | v_j(t)>
        for k in range(n_t):
            C_total[k] += np.vdot(w_j, v_evolved[k])

    # Average over sites
    C_total /= N_sites
    return C_total


def compute_one_realization(p, seed):
    """Compute transmission peak for one (sparsity, seed) pair."""
    try:
        # Build doubled system
        doubled = DoubledSYK(N_PER_SIDE, seed=seed, sparsity=p, use_sparse=True)

        # Build coupled Hamiltonian
        H_coupled = doubled.build_coupled_hamiltonian(MU)

        # Build TFD
        tfd, Z = build_tfd(doubled, BETA)

        # Compute transmission signal via Krylov
        C = transmission_krylov(H_coupled, tfd, doubled.psi_L, doubled.psi_R, t_array)

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
    print("Fix 3: N=14 Transmission Signal (Krylov method)")
    print("=" * 60)
    print(f"  N_per_side = {N_PER_SIDE} (dim = {2**N_PER_SIDE})")
    print(f"  beta = {BETA}, mu = {MU}")
    print(f"  Sparsity: {SPARSITY_VALUES}")
    print(f"  Realizations: {N_REALIZATIONS}")
    print(f"  Time: 0 to {T_MAX}, {N_T} points")
    total = len(SPARSITY_VALUES) * N_REALIZATIONS
    print(f"  Total instances: {total}")
    print()

    # Time first realization to estimate total
    print("Timing first realization (p=1.0, seed=0)...")
    t_first_start = time.time()
    first_result = compute_one_realization(1.0, 0)
    t_first = time.time() - t_first_start
    print(f"  First realization: {t_first:.1f}s")
    if first_result['success']:
        print(f"  Peak: {first_result['peak_height']:.6f} at t={first_result['peak_time']:.2f}")
    else:
        print(f"  FAILED: {first_result.get('error', '?')}")
    print()

    estimated_total = t_first * total
    print(f"  Estimated total: {estimated_total/60:.0f} min ({estimated_total/3600:.1f} hours)")

    if t_first > 300:
        print("  ABORT: Single realization exceeds 5 minutes.")
        print("  N=14 transmission is not feasible on this hardware.")
        sys.exit(1)

    if estimated_total > 4 * 3600:
        print(f"  WARNING: Estimated total ({estimated_total/3600:.1f}h) exceeds 4h budget.")
        print("  Will stop if <30% complete after 4 hours.")
    print()

    # Run all realizations
    t_start = time.time()
    all_results = [first_result]  # already have the first one
    completed = 1

    for p in SPARSITY_VALUES:
        for seed in range(N_REALIZATIONS):
            # Skip the one we already computed
            if p == 1.0 and seed == 0:
                continue

            result = compute_one_realization(p, seed)
            all_results.append(result)
            completed += 1

            elapsed = time.time() - t_start
            if completed % 5 == 0 or completed == total:
                rate = completed / (elapsed + t_first)
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] p={p:.2f} seed={seed} "
                      f"peak={result.get('peak_height', np.nan):.4f} "
                      f"({elapsed+t_first:.0f}s elapsed, ETA {eta:.0f}s)")
                sys.stdout.flush()

            # Check 4-hour budget
            total_elapsed = elapsed + t_first
            if total_elapsed > 4 * 3600 and completed / total < 0.3:
                print(f"\n  STOPPING: {total_elapsed/3600:.1f}h elapsed, "
                      f"only {completed/total*100:.0f}% complete.")
                break
        else:
            continue
        break

    total_time = time.time() - t_start + t_first
    n_success = sum(1 for r in all_results if r['success'])
    print(f"\nCompleted {completed}/{total} instances in {total_time:.0f}s "
          f"({total_time/60:.1f} min)")
    print(f"  Successes: {n_success}, Failures: {completed - n_success}")

    # Organize results
    peaks_by_sparsity = {p: [] for p in SPARSITY_VALUES}
    times_by_sparsity = {p: [] for p in SPARSITY_VALUES}

    for r in all_results:
        if r['success']:
            peaks_by_sparsity[r['sparsity']].append(r['peak_height'])
            times_by_sparsity[r['sparsity']].append(r['peak_time'])

    # Print summary
    print("\n=== Summary: Peak Height vs Sparsity (N=14, beta=8, mu=0.1) ===")
    print(f"{'p':>6} {'mean':>8} {'SEM':>8} {'std':>8} {'n':>4}")
    print("-" * 40)

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
        print(f"{p:>6.3f} {mean:>8.4f} {sem:>8.4f} {std:>8.4f} {len(peaks):>4}")

    if dense_mean is not None and dense_mean > 0:
        print(f"\n=== Ratio to Dense (p=1.0, mean={dense_mean:.4f}) ===")
        for p in SPARSITY_VALUES:
            peaks = np.array(peaks_by_sparsity[p])
            if len(peaks) == 0:
                continue
            ratio = np.mean(peaks) / dense_mean
            print(f"  p={p:.2f}: ratio = {ratio:.4f}")

    # Compare with N=10 results
    n10_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'gap3_transmission.h5')
    if os.path.exists(n10_path):
        print("\n=== N=14 vs N=10 comparison ===")
        with h5py.File(n10_path, 'r') as f:
            for p in SPARSITY_VALUES:
                key = f'p_{p:.3f}'
                if key in f:
                    n10_peaks = f[key]['peak_heights'][:]
                    n10_mean = np.mean(n10_peaks)
                    n14_peaks = np.array(peaks_by_sparsity[p])
                    if len(n14_peaks) > 0:
                        n14_mean = np.mean(n14_peaks)
                        print(f"  p={p:.2f}: N=10 mean={n10_mean:.4f}, "
                              f"N=14 mean={n14_mean:.4f}, ratio={n14_mean/n10_mean:.3f}")

    # Save to HDF5
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'fixes', 'n14_transmission.h5')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.attrs['N_per_side'] = N_PER_SIDE
        f.attrs['beta'] = BETA
        f.attrs['mu'] = MU
        f.attrs['sparsity_values'] = SPARSITY_VALUES
        f.attrs['n_realizations'] = N_REALIZATIONS
        f.attrs['t_max'] = T_MAX
        f.attrs['n_t'] = N_T
        f.attrs['method'] = 'krylov_expm_multiply'
        f.attrs['description'] = ('Fix 3: N=14 transmission peaks using Krylov time evolution, '
                                  'beta=8, mu=0.1, 30 realizations x 3 sparsity')
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['total_compute_time_s'] = total_time
        f.attrs['completed_instances'] = completed
        f.attrs['total_instances'] = total

        f.create_dataset('t_array', data=t_array)

        for p in SPARSITY_VALUES:
            grp = f.create_group(f'p_{p:.3f}')
            grp.create_dataset('peak_heights', data=np.array(peaks_by_sparsity[p]))
            grp.create_dataset('peak_times', data=np.array(times_by_sparsity[p]))

    print(f"\nSaved to {output_path}")
