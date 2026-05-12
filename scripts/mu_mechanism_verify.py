"""A6 Verification: Confirm chaos regimes at three sparsity levels.

Must pass before running the mu sweep. Checks that:
- p=1.0: <r> ~ 0.59 (GUE, chaotic)
- p=0.1: <r> ~ 0.57 (near GUE, edge of chaos)
- p=0.05: <r> ~ 0.39 (Poisson, non-chaotic)

Uses single-copy SYK eigenvalues, 50 realizations per sparsity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

from src.syk import SYKHamiltonian
from src.observables import level_spacing_ratio

N = 10
N_REALIZATIONS = 50
SPARSITY_VALUES = [1.0, 0.1, 0.05]

EXPECTED = {
    1.0:  (0.55, 0.65, 'chaotic (GUE)'),
    0.1:  (0.45, 0.65, 'near-chaotic'),
    0.05: (0.30, 0.45, 'non-chaotic (Poisson)'),
}

print("A6 Verification: Chaos Regime Check")
print("=" * 60)
print(f"  N = {N} (single-copy dim = {2**(N//2)})")
print(f"  Realizations: {N_REALIZATIONS}")
print(f"  Sparsity levels: {SPARSITY_VALUES}")
print()

t_start = time.time()
results = {}

for p in SPARSITY_VALUES:
    r_means = []
    for seed in range(N_REALIZATIONS):
        syk = SYKHamiltonian(N, seed=seed, J=1.0, sparsity=p)
        evals, _ = syk.diagonalize()
        _, r_mean = level_spacing_ratio(evals)
        r_means.append(r_mean)

    r_means = np.array(r_means)
    mean_r = np.mean(r_means)
    sem_r = np.std(r_means) / np.sqrt(len(r_means))
    std_r = np.std(r_means)

    lo, hi, label = EXPECTED[p]
    passed = lo <= mean_r <= hi

    results[p] = {
        'r_means': r_means,
        'mean': mean_r,
        'sem': sem_r,
        'std': std_r,
        'passed': passed,
    }

    status = "PASS" if passed else "FAIL"
    print(f"  p={p:.2f}: <r> = {mean_r:.4f} +/- {sem_r:.4f} "
          f"(std={std_r:.4f})  [{status}] expected {label} ({lo:.2f}-{hi:.2f})")

elapsed = time.time() - t_start
print(f"\nCompleted in {elapsed:.1f}s")

all_passed = all(r['passed'] for r in results.values())
print()
if all_passed:
    print("ALL CHECKS PASSED. Chaos regimes confirmed. Safe to proceed with mu sweep.")
else:
    print("VERIFICATION FAILED. Do NOT proceed with mu sweep.")
    for p, r in results.items():
        if not r['passed']:
            lo, hi, label = EXPECTED[p]
            print(f"  p={p}: <r>={r['mean']:.4f} outside expected range [{lo}, {hi}]")
    sys.exit(1)
