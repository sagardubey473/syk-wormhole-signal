"""Gap 3: Level spacing ratio computation at proper statistics.

N=10 (GUE, N mod 8 = 2) and N=14 (GUE, N mod 8 = 6).
50 realizations per (N, p) point, 9 sparsity values.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import time
from src.syk import SYKHamiltonian
from src.observables import level_spacing_ratio

# Parameters
N_VALUES = [10, 14]
SPARSITY_VALUES = [1.0, 0.5, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.02]
N_REALIZATIONS = 50
BETA = 8.0

print(f"Gap 3: Level spacing ratio computation")
print(f"  N values: {N_VALUES}")
print(f"  Sparsity values: {SPARSITY_VALUES}")
print(f"  Realizations: {N_REALIZATIONS}")
print(f"  Total instances: {len(N_VALUES) * len(SPARSITY_VALUES) * N_REALIZATIONS}")
print()

results = {}

for N in N_VALUES:
    dim_single = 2 ** (N // 2)
    print(f"N={N} (dim={dim_single}):")

    for p in SPARSITY_VALUES:
        t0 = time.time()
        r_means = []
        all_r_values = []

        for seed in range(N_REALIZATIONS):
            syk = SYKHamiltonian(N, seed=seed, sparsity=p, use_sparse=False)
            evals, _ = syk.diagonalize()
            r_vals, r_mean = level_spacing_ratio(evals)
            r_means.append(r_mean)
            all_r_values.extend(r_vals.tolist())

        r_means = np.array(r_means)
        mean_r = np.mean(r_means)
        std_r = np.std(r_means)
        sem_r = std_r / np.sqrt(N_REALIZATIONS)

        dt = time.time() - t0
        print(f"  p={p:.3f}: <r>={mean_r:.4f} +/- {sem_r:.4f} (std={std_r:.4f}) [{dt:.1f}s]")

        results[f"N{N}_p{p:.3f}_r_means"] = r_means
        results[f"N{N}_p{p:.3f}_r_all"] = np.array(all_r_values)
    print()

# Save to HDF5
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'gap3_level_spacing.h5')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with h5py.File(output_path, 'w') as f:
    # Metadata
    f.attrs['N_values'] = N_VALUES
    f.attrs['sparsity_values'] = SPARSITY_VALUES
    f.attrs['n_realizations'] = N_REALIZATIONS
    f.attrs['beta'] = BETA
    f.attrs['seeds'] = list(range(N_REALIZATIONS))
    f.attrs['description'] = 'Gap 3: Level spacing ratios at 50 realizations for GUE systems'
    f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    for key, val in results.items():
        f.create_dataset(key, data=val)

print(f"\nSaved to {output_path}")
print("\n=== Summary Table ===")
print(f"{'N':>4} {'p':>6} {'<r>':>8} {'SEM':>8} {'Class':>12}")
print("-" * 45)
for N in N_VALUES:
    for p in SPARSITY_VALUES:
        r_means = results[f"N{N}_p{p:.3f}_r_means"]
        mean_r = np.mean(r_means)
        sem_r = np.std(r_means) / np.sqrt(N_REALIZATIONS)
        if mean_r > 0.55:
            cls = "GUE"
        elif mean_r > 0.48:
            cls = "GOE"
        elif mean_r > 0.42:
            cls = "Trans."
        else:
            cls = "Poisson"
        print(f"{N:>4} {p:>6.3f} {mean_r:>8.4f} {sem_r:>8.4f} {cls:>12}")
    print()
