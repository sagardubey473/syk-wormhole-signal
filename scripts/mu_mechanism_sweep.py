"""Mu mechanism sweep: H(mu, p) across coupling and sparsity.

For each (p, seed): build DoubledSYK and TFD once, then sweep mu.
This avoids redundant Hamiltonian construction and TFD computation.

Output: data/research/mu_mechanism.h5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import time

from src.doubled import DoubledSYK
from src.tfd import build_tfd
from src.wormhole import transmission_signal, extract_peak
from src.syk import SYKHamiltonian
from src.observables import level_spacing_ratio

# Parameters
N_PER_SIDE = 10
BETA = 8.0
SPARSITY_VALUES = [1.0, 0.1, 0.05]
MU_VALUES = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
N_REALIZATIONS = 50
T_MAX = 50.0
N_T = 200

t_array = np.linspace(0, T_MAX, N_T)

total_instances = len(SPARSITY_VALUES) * len(MU_VALUES) * N_REALIZATIONS
total_base = len(SPARSITY_VALUES) * N_REALIZATIONS  # unique (p, seed) pairs

print("Mu Mechanism Sweep")
print("=" * 70)
print(f"  N_per_side = {N_PER_SIDE} (doubled dim = {2**N_PER_SIDE})")
print(f"  beta = {BETA}")
print(f"  Sparsity: {SPARSITY_VALUES}")
print(f"  Mu values: {MU_VALUES}")
print(f"  Realizations: {N_REALIZATIONS}")
print(f"  Time: 0 to {T_MAX}, {N_T} points")
print(f"  Total instances: {total_instances}")
print(f"  Unique (p, seed) pairs: {total_base} (build once, sweep {len(MU_VALUES)} mu)")
print()

# Storage
peak_heights = np.full((len(SPARSITY_VALUES), len(MU_VALUES), N_REALIZATIONS), np.nan)
peak_times = np.full((len(SPARSITY_VALUES), len(MU_VALUES), N_REALIZATIONS), np.nan)
peak_fwhms = np.full((len(SPARSITY_VALUES), len(MU_VALUES), N_REALIZATIONS), np.nan)
r_values = np.full((len(SPARSITY_VALUES), N_REALIZATIONS), np.nan)

# Time first instance
print("Timing first (p=1.0, seed=0, mu=0.1)...")
t0 = time.time()
doubled_test = DoubledSYK(N_PER_SIDE, seed=0, sparsity=1.0)
tfd_test, _ = build_tfd(doubled_test, BETA)
H_test = doubled_test.build_coupled_hamiltonian(0.1)
C_test = transmission_signal(H_test, tfd_test, doubled_test.psi_L, doubled_test.psi_R,
                              t_array, use_eigen=True)
ph, pt, fw = extract_peak(C_test, t_array)
t_first = time.time() - t0
print(f"  Time: {t_first:.2f}s (peak={ph:.4f} at t={pt:.2f})")

# Estimate: building doubled+TFD is ~60% of cost, each mu sweep is ~40%
# Per base pair: t_build + n_mu * t_mu
# Rough: t_build ~ 0.6*t_first, t_mu ~ 0.4*t_first
t_per_base = t_first * 0.6
t_per_mu = t_first * 0.4
estimated = total_base * t_per_base + total_instances * t_per_mu
print(f"  Estimated total: {estimated/60:.0f} min ({estimated/3600:.1f} hours)")
print()

# Main loop
t_start = time.time()
completed = 0
failures = 0

for ip, p in enumerate(SPARSITY_VALUES):
    for seed in range(N_REALIZATIONS):
        try:
            # Build doubled system and TFD once
            doubled = DoubledSYK(N_PER_SIDE, seed=seed, sparsity=p)
            tfd, Z = build_tfd(doubled, BETA)

            # Level spacing from single-copy SYK
            syk_single = SYKHamiltonian(N_PER_SIDE, seed=seed, J=1.0, sparsity=p)
            evals_single, _ = syk_single.diagonalize()
            _, r_mean = level_spacing_ratio(evals_single)
            r_values[ip, seed] = r_mean

            # Sweep mu
            for imu, mu in enumerate(MU_VALUES):
                H_coupled = doubled.build_coupled_hamiltonian(mu)
                C = transmission_signal(H_coupled, tfd, doubled.psi_L, doubled.psi_R,
                                        t_array, use_eigen=True)
                ph, pt, fw = extract_peak(C, t_array)
                peak_heights[ip, imu, seed] = ph
                peak_times[ip, imu, seed] = pt
                peak_fwhms[ip, imu, seed] = fw
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
            latest_peak = peak_heights[ip, 3, seed]  # mu=0.10
            print(f"  [{base_done}/{total_base}] p={p:.2f} seed={seed} "
                  f"peak(mu=0.1)={latest_peak:.4f} <r>={r_mean:.4f} "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            sys.stdout.flush()

total_time = time.time() - t_start
print(f"\nCompleted {completed}/{total_instances} in {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Failures: {failures}")

# Save to HDF5
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'research')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'mu_mechanism.h5')

with h5py.File(output_path, 'w') as f:
    # Metadata
    f.attrs['N_per_side'] = N_PER_SIDE
    f.attrs['beta'] = BETA
    f.attrs['sparsity_values'] = SPARSITY_VALUES
    f.attrs['mu_values'] = MU_VALUES
    f.attrs['n_realizations'] = N_REALIZATIONS
    f.attrs['t_max'] = T_MAX
    f.attrs['n_t'] = N_T
    f.attrs['total_compute_time_s'] = total_time
    f.attrs['total_instances'] = total_instances
    f.attrs['failures'] = failures
    f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    f.attrs['description'] = (
        'Mu mechanism sweep: transmission peak vs coupling strength mu '
        'at three sparsity levels (chaotic, edge, non-chaotic). '
        'Tests whether mu controls signal independently of internal chaos.'
    )

    # Raw data arrays: shape (n_sparsity, n_mu, n_realizations)
    f.create_dataset('peak_heights', data=peak_heights)
    f.create_dataset('peak_times', data=peak_times)
    f.create_dataset('peak_fwhms', data=peak_fwhms)
    f.create_dataset('r_values', data=r_values)  # shape (n_sparsity, n_realizations)
    f.create_dataset('t_array', data=t_array)

print(f"\nSaved to {output_path}")

# Print summary
print("\n=== Summary: Peak Height H(mu, p) ===")
print(f"{'mu':>6}", end='')
for p in SPARSITY_VALUES:
    print(f"  p={p:.2f} (mean+/-SEM)", end='')
print()
print("-" * 70)

for imu, mu in enumerate(MU_VALUES):
    print(f"{mu:>6.2f}", end='')
    for ip, p in enumerate(SPARSITY_VALUES):
        vals = peak_heights[ip, imu, :]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            m = np.mean(valid)
            s = np.std(valid) / np.sqrt(len(valid))
            print(f"  {m:.4f}+/-{s:.4f}     ", end='')
        else:
            print(f"  {'N/A':>18}", end='')
    print()

print("\n=== Level Spacing Verification ===")
for ip, p in enumerate(SPARSITY_VALUES):
    vals = r_values[ip, :]
    valid = vals[~np.isnan(vals)]
    print(f"  p={p:.2f}: <r> = {np.mean(valid):.4f} +/- {np.std(valid)/np.sqrt(len(valid)):.4f}")
