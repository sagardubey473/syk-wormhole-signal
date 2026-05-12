"""Recompute FWHM with corrected extract_peak for all 1200 instances.

Reuses the same seeds and parameters as the original mu_mechanism_sweep.py.
Optimization: eigendecompose H(mu) once per mu, then vectorize C(t) computation.

Output: data/research/mu_mechanism_corrected_fwhm.h5
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

# Parameters (must match original sweep)
N_PER_SIDE = 10
BETA = 8.0
SPARSITY_VALUES = [1.0, 0.1, 0.05]
MU_VALUES = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
N_REALIZATIONS = 50
T_MAX = 50.0
N_T = 200

t_array = np.linspace(0, T_MAX, N_T)

total_instances = len(SPARSITY_VALUES) * len(MU_VALUES) * N_REALIZATIONS
total_base = len(SPARSITY_VALUES) * N_REALIZATIONS

print("FWHM Recomputation with Corrected extract_peak")
print("=" * 60)
print(f"  N_per_side = {N_PER_SIDE} (dim = {2**N_PER_SIDE})")
print(f"  beta = {BETA}")
print(f"  Sparsity: {SPARSITY_VALUES}")
print(f"  Mu values: {MU_VALUES}")
print(f"  Realizations: {N_REALIZATIONS}")
print(f"  Total instances: {total_instances}")
print(f"  Unique (p, seed) pairs: {total_base}")
print()


def transmission_signal_fast(evals, evecs, tfd, psi_L, psi_R, t_array, sites=None):
    """Optimized C(t) using pre-computed eigendecomposition.

    Vectorized over time using einsum.
    """
    N_per_side = len(psi_L)
    if sites is None:
        sites = list(range(N_per_side))

    tfd_eig = evecs.conj().T @ tfd
    dE = np.subtract.outer(evals, evals)  # (dim, dim)

    # Precompute phase matrix for all times: (n_t, dim, dim)
    # For memory: dim=1024, n_t=200 -> 200*1024*1024*16 bytes = ~3.2 GB (too much)
    # Instead, accumulate per-site contribution

    C = np.zeros(len(t_array), dtype=complex)

    for j in sites:
        pL = psi_L[j].toarray() if sparse.issparse(psi_L[j]) else psi_L[j]
        pR = psi_R[j].toarray() if sparse.issparse(psi_R[j]) else psi_R[j]

        ket = pL @ tfd
        ket_eig = evecs.conj().T @ ket
        R_eig = evecs.conj().T @ pR @ evecs

        A = np.outer(tfd_eig.conj(), ket_eig) * R_eig  # (dim, dim)

        # Vectorize over time: C(t) = sum_{m,n} A_{mn} exp(i*dE_{mn}*t)
        # Flatten A and dE, then use outer product with t
        A_flat = A.ravel()  # (dim^2,)
        dE_flat = dE.ravel()  # (dim^2,)

        # Keep only significant elements to speed up
        mask = np.abs(A_flat) > 1e-15
        A_sig = A_flat[mask]
        dE_sig = dE_flat[mask]

        # phases: (n_t, n_sig)
        phases = np.exp(1j * np.outer(t_array, dE_sig))
        C += phases @ A_sig

    C /= len(sites)
    return C


# Storage
corrected_fwhm = np.full((len(SPARSITY_VALUES), len(MU_VALUES), N_REALIZATIONS), np.nan)
# Also store peak heights and times for cross-validation
corrected_heights = np.full((len(SPARSITY_VALUES), len(MU_VALUES), N_REALIZATIONS), np.nan)
corrected_times = np.full((len(SPARSITY_VALUES), len(MU_VALUES), N_REALIZATIONS), np.nan)

t_start = time.time()
completed = 0
failures = 0

for ip, p in enumerate(SPARSITY_VALUES):
    for seed in range(N_REALIZATIONS):
        try:
            # Build doubled system and TFD once per (p, seed)
            doubled = DoubledSYK(N_PER_SIDE, seed=seed, sparsity=p)
            tfd, Z = build_tfd(doubled, BETA)

            # Sweep mu
            for imu, mu in enumerate(MU_VALUES):
                H_coupled = doubled.build_coupled_hamiltonian(mu)

                # Eigendecompose
                if sparse.issparse(H_coupled):
                    H_dense = H_coupled.toarray()
                else:
                    H_dense = H_coupled
                H_herm = 0.5 * (H_dense + H_dense.conj().T)
                evals, evecs = np.linalg.eigh(H_herm)

                # Compute C(t)
                C = transmission_signal_fast(evals, evecs, tfd,
                                             doubled.psi_L, doubled.psi_R,
                                             t_array)

                # Extract peak with corrected FWHM
                ph, pt, fw = extract_peak(C, t_array)
                corrected_heights[ip, imu, seed] = ph
                corrected_times[ip, imu, seed] = pt
                corrected_fwhm[ip, imu, seed] = fw
                completed += 1

        except Exception as e:
            print(f"  ERROR: p={p}, seed={seed}: {e}")
            failures += 1
            completed += len(MU_VALUES)
            continue

        # Progress
        base_done = ip * N_REALIZATIONS + seed + 1
        if base_done % 10 == 0 or base_done == total_base:
            elapsed = time.time() - t_start
            rate = base_done / elapsed
            eta = (total_base - base_done) / rate if rate > 0 else 0
            latest_h = corrected_heights[ip, 3, seed]  # mu=0.10
            latest_fw = corrected_fwhm[ip, 3, seed]
            print(f"  [{base_done}/{total_base}] p={p:.2f} seed={seed} "
                  f"H(0.1)={latest_h:.4f} FWHM(0.1)={latest_fw:.2f} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            sys.stdout.flush()

total_time = time.time() - t_start
print(f"\nCompleted {completed}/{total_instances} in {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Failures: {failures}")

# Save to HDF5
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'research')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'mu_mechanism_corrected_fwhm.h5')

with h5py.File(output_path, 'w') as f:
    f.attrs['N_per_side'] = N_PER_SIDE
    f.attrs['beta'] = BETA
    f.attrs['sparsity_values'] = SPARSITY_VALUES
    f.attrs['mu_values'] = MU_VALUES
    f.attrs['n_realizations'] = N_REALIZATIONS
    f.attrs['t_max'] = T_MAX
    f.attrs['n_t'] = N_T
    f.attrs['total_compute_time_s'] = total_time
    f.attrs['failures'] = failures
    f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    f.attrs['description'] = (
        'Corrected FWHM recomputation using fixed extract_peak. '
        'FWHM now measured by left/right search from peak with '
        'linear interpolation. NaN if signal does not drop below '
        'half-max on either side.'
    )

    f.create_dataset('corrected_fwhm', data=corrected_fwhm)
    f.create_dataset('corrected_heights', data=corrected_heights)
    f.create_dataset('corrected_times', data=corrected_times)
    f.create_dataset('t_array', data=t_array)

print(f"\nSaved to {output_path}")

# Cross-validate peak heights against original
original_path = os.path.join(output_dir, 'mu_mechanism.h5')
if os.path.exists(original_path):
    with h5py.File(original_path, 'r') as f:
        orig_heights = f['peak_heights'][:]
        orig_fwhm = f['peak_fwhms'][:]

    # Peak heights should match (same computation)
    diff = np.abs(corrected_heights - orig_heights)
    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff)
    print(f"\n=== Cross-validation: Peak Heights ===")
    print(f"  Max |diff|: {max_diff:.6e}")
    print(f"  Mean |diff|: {mean_diff:.6e}")
    if max_diff < 1e-6:
        print("  PASS: Heights match original (same seeds)")
    else:
        print("  WARNING: Heights differ - check seed consistency")

    print(f"\n=== FWHM Comparison: Old vs Corrected ===")
    print(f"{'mu':>6}", end='')
    for p_val in SPARSITY_VALUES:
        print(f"  p={p_val:.2f} old->new", end='')
    print()
    print("-" * 80)

    for imu, mu in enumerate(MU_VALUES):
        print(f"{mu:>6.2f}", end='')
        for ip in range(len(SPARSITY_VALUES)):
            old_vals = orig_fwhm[ip, imu, :]
            new_vals = corrected_fwhm[ip, imu, :]
            old_m = np.nanmean(old_vals)
            new_m = np.nanmean(new_vals)
            n_nan = np.sum(np.isnan(new_vals))
            print(f"  {old_m:5.1f}->{new_m:5.1f} ({n_nan}NaN)", end='')
        print()
