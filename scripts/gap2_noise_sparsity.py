"""Gap 2: Noise × Sparsity 2D cross-cut.

N_per_side=8 (dim=256), beta=8, mu=0.1.
Sparsity p in {1.0, 0.3, 0.1, 0.05} (4 points).
Noise gamma in {0, 0.001, 0.003, 0.01, 0.03, 0.1} (6 points).
30 realizations per (p, gamma) point.

Uses optimized dephasing-specific Lindblad solver (element-wise decay
instead of explicit operator products). ~230x faster than generic Lindblad.

Note: N=10 Lindblad (dim=1024) is computationally prohibitive for 30+ realizations.
N=8 (dim=256) is used following the same system size as the previous RQ3 computation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import time
from scipy.integrate import solve_ivp
from scipy import sparse

from src.doubled import DoubledSYK
from src.tfd import build_tfd
from src.wormhole import extract_peak

# Parameters
N_PER_SIDE = 8
BETA = 8.0
MU = 0.1
SPARSITY_VALUES = [1.0, 0.3, 0.1, 0.05]
GAMMA_VALUES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
N_REALIZATIONS = 30
T_MAX = 30.0
N_T = 80
N_SITES = 8  # average over all Majorana sites

t_array = np.linspace(0, T_MAX, N_T)


def make_hamming_decay_matrix(n_qubits, gamma):
    """Precompute decay rates for dephasing.

    For dephasing with L_k = sqrt(gamma) Z_k, the dissipative term is:
    D[rho]_{ab} = -2 * gamma * hamming_dist(a,b) * rho_{ab}

    This replaces the full Lindblad operator machinery with a single
    element-wise multiplication.
    """
    dim = 2 ** n_qubits
    rates = np.zeros((dim, dim))
    for a in range(dim):
        for b in range(dim):
            rates[a, b] = -2.0 * gamma * bin(a ^ b).count('1')
    return rates


def fast_dephasing_rhs(t, rho_vec, H, decay_rates, dim):
    """Fast RHS for dephasing Lindblad equation.

    drho/dt = -i[H, rho] + decay_rates * rho  (element-wise)
    """
    rho = rho_vec.reshape(dim, dim)
    drho = -1j * (H @ rho - rho @ H) + decay_rates * rho
    return drho.ravel()


def transmission_noiseless_optimized(H_coupled, tfd, psi_L, psi_R, t_array, n_sites):
    """Optimized noiseless transmission using eigendecomposition."""
    if sparse.issparse(H_coupled):
        H_dense = H_coupled.toarray()
    else:
        H_dense = H_coupled
    H_herm = 0.5 * (H_dense + H_dense.conj().T)
    evals, evecs = np.linalg.eigh(H_herm)

    tfd_eig = evecs.conj().T @ tfd
    dE = np.subtract.outer(evals, evals)

    A_total = np.zeros_like(dE, dtype=complex)
    for j in range(n_sites):
        pL = psi_L[j].toarray() if sparse.issparse(psi_L[j]) else psi_L[j]
        pR = psi_R[j].toarray() if sparse.issparse(psi_R[j]) else psi_R[j]
        ket = pL @ tfd
        ket_eig = evecs.conj().T @ ket
        R_eig = evecs.conj().T @ pR @ evecs
        A_total += np.outer(tfd_eig.conj(), ket_eig) * R_eig
    A_total /= n_sites

    C = np.zeros(len(t_array), dtype=complex)
    for idx, t in enumerate(t_array):
        C[idx] = np.sum(A_total * np.exp(1j * dE * t))
    return C


def transmission_noisy_optimized(H_coupled, tfd, psi_L, psi_R,
                                  t_array, n_sites, gamma):
    """Noisy transmission using optimized dephasing Lindblad."""
    if sparse.issparse(H_coupled):
        H_dense = H_coupled.toarray()
    else:
        H_dense = H_coupled
    H_herm = 0.5 * (H_dense + H_dense.conj().T)
    dim = len(tfd)
    n_qubits = int(np.log2(dim))

    decay_rates = make_hamming_decay_matrix(n_qubits, gamma)
    rho0 = np.outer(tfd, tfd.conj())

    pL_dense = [op.toarray() if sparse.issparse(op) else op for op in psi_L[:n_sites]]
    pR_dense = [op.toarray() if sparse.issparse(op) else op for op in psi_R[:n_sites]]

    C = np.zeros(len(t_array), dtype=complex)
    for j in range(n_sites):
        rho_init = pL_dense[j] @ rho0
        sol = solve_ivp(fast_dephasing_rhs, (t_array[0], t_array[-1]),
                        rho_init.ravel(), t_eval=t_array, method='RK45',
                        args=(H_herm, decay_rates, dim),
                        rtol=1e-6, atol=1e-8)
        rho_t = sol.y.T.reshape(len(t_array), dim, dim)
        for idx in range(len(t_array)):
            C[idx] += np.trace(pR_dense[j] @ rho_t[idx])

    C /= n_sites
    return C


def compute_one_realization(args):
    """Compute noisy transmission peak for one (sparsity, gamma, seed) triple."""
    p, gamma, seed = args
    try:
        doubled = DoubledSYK(N_PER_SIDE, seed=seed, sparsity=p, use_sparse=None)
        H_coupled = doubled.build_coupled_hamiltonian(MU)
        tfd, Z = build_tfd(doubled, BETA)

        if gamma == 0.0:
            C = transmission_noiseless_optimized(
                H_coupled, tfd, doubled.psi_L, doubled.psi_R, t_array, N_SITES)
        else:
            C = transmission_noisy_optimized(
                H_coupled, tfd, doubled.psi_L, doubled.psi_R,
                t_array, N_SITES, gamma)

        peak_height, peak_time, fwhm = extract_peak(C, t_array)

        return {
            'seed': seed, 'sparsity': p, 'gamma': gamma,
            'peak_height': peak_height, 'peak_time': peak_time,
            'fwhm': fwhm, 'success': True
        }
    except Exception as e:
        return {
            'seed': seed, 'sparsity': p, 'gamma': gamma,
            'peak_height': np.nan, 'peak_time': np.nan,
            'fwhm': np.nan, 'success': False, 'error': str(e)
        }


if __name__ == '__main__':
    print(f"Gap 2: Noise × Sparsity cross-cut")
    print(f"  N_per_side = {N_PER_SIDE} (dim = {2**N_PER_SIDE})")
    print(f"  beta = {BETA}, mu = {MU}")
    print(f"  Sparsity values: {SPARSITY_VALUES}")
    print(f"  Gamma values: {GAMMA_VALUES}")
    print(f"  Realizations: {N_REALIZATIONS}")
    print(f"  Time: 0 to {T_MAX}, {N_T} points, {N_SITES} sites")
    total = len(SPARSITY_VALUES) * len(GAMMA_VALUES) * N_REALIZATIONS
    print(f"  Total instances: {total}")
    print()

    tasks = []
    for p in SPARSITY_VALUES:
        for gamma in GAMMA_VALUES:
            for seed in range(N_REALIZATIONS):
                tasks.append((p, gamma, seed))

    t_start = time.time()
    all_results = []

    for i, task in enumerate(tasks):
        result = compute_one_realization(task)
        all_results.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            p_curr, g_curr, _ = task
            print(f"  [{i+1}/{total}] p={p_curr:.2f},g={g_curr:.4f} "
                  f"{elapsed:.0f}s, ETA {eta:.0f}s", flush=True)

    total_time = time.time() - t_start
    print(f"\nAll {total} instances complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Organize results
    peaks = {}
    times = {}
    for p in SPARSITY_VALUES:
        for gamma in GAMMA_VALUES:
            key = (p, gamma)
            peaks[key] = []
            times[key] = []

    n_failures = 0
    for r in all_results:
        if r['success']:
            key = (r['sparsity'], r['gamma'])
            peaks[key].append(r['peak_height'])
            times[key].append(r['peak_time'])
        else:
            n_failures += 1
            print(f"  FAILURE: seed={r['seed']}, p={r['sparsity']}, "
                  f"gamma={r['gamma']}: {r.get('error','?')}")

    if n_failures > 0:
        print(f"\n  WARNING: {n_failures} failures out of {total}")

    # Save to HDF5
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'gap2_noise_sparsity.h5')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.attrs['N_per_side'] = N_PER_SIDE
        f.attrs['beta'] = BETA
        f.attrs['mu'] = MU
        f.attrs['sparsity_values'] = SPARSITY_VALUES
        f.attrs['gamma_values'] = GAMMA_VALUES
        f.attrs['n_realizations'] = N_REALIZATIONS
        f.attrs['t_max'] = T_MAX
        f.attrs['n_t'] = N_T
        f.attrs['n_sites'] = N_SITES
        f.attrs['seeds'] = list(range(N_REALIZATIONS))
        f.attrs['description'] = ('Gap 2: Noise x Sparsity 2D cross-cut, '
                                  f'N={N_PER_SIDE}, beta={BETA}, mu={MU}, '
                                  f'{N_REALIZATIONS} realizations, optimized dephasing')
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['total_compute_time_s'] = total_time

        for p in SPARSITY_VALUES:
            grp = f.create_group(f'p_{p:.3f}')
            for gamma in GAMMA_VALUES:
                key = (p, gamma)
                sub = grp.create_group(f'gamma_{gamma:.4f}')
                sub.create_dataset('peak_heights', data=np.array(peaks[key]))
                sub.create_dataset('peak_times', data=np.array(times[key]))

    print(f"\nSaved to {output_path}")

    # Print summary table
    print("\n=== Noisy Peak Height: mean (SEM) ===")
    header = f"{'p \\\\ gamma':>12}"
    for gamma in GAMMA_VALUES:
        header += f" {gamma:>10.4f}"
    print(header)
    print("-" * (12 + 11 * len(GAMMA_VALUES)))

    for p in SPARSITY_VALUES:
        row = f"{p:>12.3f}"
        for gamma in GAMMA_VALUES:
            key = (p, gamma)
            arr = np.array(peaks[key])
            if len(arr) > 0:
                row += f" {np.mean(arr):>5.3f}({np.std(arr)/np.sqrt(len(arr)):>4.3f})"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # Compute 50% degradation thresholds
    print("\n=== 50% Degradation Threshold gamma*(p) ===")
    for p in SPARSITY_VALUES:
        noiseless = np.array(peaks[(p, 0.0)])
        if len(noiseless) == 0:
            continue
        noiseless_mean = np.mean(noiseless)
        half_target = noiseless_mean * 0.5

        gamma_star = None
        for i, gamma in enumerate(GAMMA_VALUES[1:], 1):
            noisy_mean = np.mean(np.array(peaks[(p, gamma)]))
            if noisy_mean < half_target:
                prev_gamma = GAMMA_VALUES[i-1]
                prev_mean = np.mean(np.array(peaks[(p, prev_gamma)]))
                if prev_mean > half_target:
                    frac = (prev_mean - half_target) / (prev_mean - noisy_mean)
                    gamma_star = prev_gamma * (gamma / prev_gamma) ** frac
                else:
                    gamma_star = prev_gamma
                break

        if gamma_star is not None:
            print(f"  p={p:.3f}: gamma* = {gamma_star:.4f}")
        else:
            print(f"  p={p:.3f}: gamma* > {GAMMA_VALUES[-1]}")

    # Hardware comparison
    print("\n=== Hardware Comparison ===")
    print("Assuming J = 10 MHz (superconducting qubit scale):")
    print("  T2 = 50 us → gamma_phi = 1/(T2*J) = 1/(50e-6 * 10e6) = 0.002 J")
    print("  T2 = 100 us → gamma_phi = 0.001 J")
    print("  Per Trotter step: gamma_eff ~ gamma_phi * dt, dt ~ 0.1/J → gamma_eff ~ 1e-4")
    print("  For 100 Trotter steps: cumulative gamma ~ 0.01-0.02")
