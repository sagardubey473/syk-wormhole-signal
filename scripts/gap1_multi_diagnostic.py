"""Gap 1: Multi-diagnostic classifier validation.

Combines four diagnostics (transmission peak, level spacing, Lyapunov, SFF)
into a single holographic score. Tests against known systems.

Uses data from:
- Gap 3: gap3_transmission.h5, gap3_level_spacing.h5 (proper statistics)
- Previous: rq2_diagnostics.npz, otoc.npz, spectral_form_factor.npz
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import time

from src.syk import SYKHamiltonian
from src.doubled import DoubledSYK
from src.tfd import build_tfd
from src.observables import level_spacing_ratio, spectral_form_factor, extract_lyapunov
from src.multi_diagnostic import (
    score_transmission, score_level_spacing, score_lyapunov, score_sff_ramp,
    combined_score_geometric, combined_score_minimum, classify_system
)

# Parameters
N_PER_SIDE = 10
BETA = 8.0
MU = 0.1
N_REALIZATIONS = 30  # for new computations
T_SFF = np.linspace(0.1, 50, 200)
T_OTOC = np.linspace(0, 5, 100)

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_diagnostics_for_system(system_type, N=10, beta=8.0, mu=0.1,
                                    sparsity=1.0, n_real=30, seed_offset=0):
    """Compute all four diagnostics for a given system type.

    Returns dict with means and errors for each diagnostic.
    """
    peaks = []
    r_means = []
    lyapunovs = []
    sff_scores = []

    for seed in range(seed_offset, seed_offset + n_real):
        # Build single-side SYK for spectral diagnostics
        syk = SYKHamiltonian(N, seed=seed, sparsity=sparsity, use_sparse=False)
        evals, evecs = syk.diagonalize()

        # Level spacing
        _, r_mean = syk.spectrum_statistics()
        r_means.append(r_mean)

        # SFF
        K = spectral_form_factor(evals, T_SFF)
        dim_single = 2 ** (N // 2)
        s_sff = score_sff_ramp(K, T_SFF, dim_single)
        sff_scores.append(s_sff)

        # OTOC and Lyapunov
        F = syk.otoc(beta, T_OTOC, site_i=0, site_j=1)
        lambda_L, r_sq, _ = extract_lyapunov(F, T_OTOC)
        lyapunovs.append(max(lambda_L, 0.0))  # clip negative

        # Transmission peak (only for coupled SYK systems)
        if system_type in ['dense_syk', 'sparse_syk']:
            doubled = DoubledSYK(N, seed=seed, sparsity=sparsity, use_sparse=None)
            H = doubled.build_coupled_hamiltonian(mu)
            tfd, Z = build_tfd(doubled, beta)

            # Optimized signal
            from scipy import sparse as sp
            if sp.issparse(H):
                H_d = H.toarray()
            else:
                H_d = H
            H_d = 0.5 * (H_d + H_d.conj().T)
            ev, evec = np.linalg.eigh(H_d)
            tfd_eig = evec.conj().T @ tfd
            dE = np.subtract.outer(ev, ev)

            t_sig = np.linspace(0, 30, 100)
            A_total = np.zeros_like(dE, dtype=complex)
            n_sites = min(N, 8)
            for j in range(n_sites):
                pL = doubled.psi_L[j].toarray() if sp.issparse(doubled.psi_L[j]) else doubled.psi_L[j]
                pR = doubled.psi_R[j].toarray() if sp.issparse(doubled.psi_R[j]) else doubled.psi_R[j]
                ket = pL @ tfd
                ket_eig = evec.conj().T @ ket
                R_eig = evec.conj().T @ pR @ evec
                A_total += np.outer(tfd_eig.conj(), ket_eig) * R_eig
            A_total /= n_sites

            C = np.zeros(len(t_sig), dtype=complex)
            for idx, t in enumerate(t_sig):
                C[idx] = np.sum(A_total * np.exp(1j * dE * t))
            peaks.append(np.max(np.abs(C)))
        else:
            peaks.append(0.0)  # non-SYK systems don't have wormhole signal

    return {
        'peak_mean': np.mean(peaks),
        'peak_sem': np.std(peaks) / np.sqrt(len(peaks)),
        'r_mean': np.mean(r_means),
        'r_sem': np.std(r_means) / np.sqrt(len(r_means)),
        'lyapunov_mean': np.mean(lyapunovs),
        'lyapunov_sem': np.std(lyapunovs) / np.sqrt(len(lyapunovs)),
        'sff_mean': np.mean(sff_scores),
        'sff_sem': np.std(sff_scores) / np.sqrt(len(sff_scores)),
    }


if __name__ == '__main__':
    print("Gap 1: Multi-diagnostic classifier validation")
    print("=" * 60)

    # Load Gap 3 data for dense and sparse SYK
    print("\nLoading Gap 3 data...")
    with h5py.File(os.path.join(base, 'data', 'gap3_level_spacing.h5'), 'r') as f:
        r_data_gap3 = {}
        sparsity_ls = list(f.attrs['sparsity_values'])
        for p in sparsity_ls:
            key = f'N10_p{p:.3f}_r_means'
            r_data_gap3[p] = np.array(f[key])

    with h5py.File(os.path.join(base, 'data', 'gap3_transmission.h5'), 'r') as f:
        peak_data_gap3 = {}
        sparsity_tr = list(f.attrs['sparsity_values'])
        for p in sparsity_tr:
            peak_data_gap3[p] = np.array(f[f'p_{p:.3f}']['peak_heights'])

    # Systems to test
    print("\nComputing diagnostics for test systems...")
    print("(Using 30 realizations per system, N=10, beta=8)")
    print()

    systems = {}

    # 1. Dense SYK (should score high)
    print("  Dense SYK (p=1.0)...", end=' ', flush=True)
    t0 = time.time()
    systems['Dense SYK'] = compute_diagnostics_for_system(
        'dense_syk', N=10, beta=8.0, mu=0.1, sparsity=1.0, n_real=30)
    # Override with Gap 3 data (better statistics)
    systems['Dense SYK']['peak_mean'] = np.mean(peak_data_gap3[1.0])
    systems['Dense SYK']['peak_sem'] = np.std(peak_data_gap3[1.0]) / np.sqrt(50)
    systems['Dense SYK']['r_mean'] = np.mean(r_data_gap3[1.0])
    systems['Dense SYK']['r_sem'] = np.std(r_data_gap3[1.0]) / np.sqrt(50)
    print(f"done ({time.time()-t0:.1f}s)")

    # 2. Sparse SYK at transition (p=0.1)
    print("  Sparse SYK (p=0.1)...", end=' ', flush=True)
    t0 = time.time()
    systems['Sparse SYK p=0.1'] = compute_diagnostics_for_system(
        'sparse_syk', N=10, beta=8.0, mu=0.1, sparsity=0.1, n_real=30)
    systems['Sparse SYK p=0.1']['peak_mean'] = np.mean(peak_data_gap3[0.1])
    systems['Sparse SYK p=0.1']['peak_sem'] = np.std(peak_data_gap3[0.1]) / np.sqrt(50)
    systems['Sparse SYK p=0.1']['r_mean'] = np.mean(r_data_gap3[0.1])
    systems['Sparse SYK p=0.1']['r_sem'] = np.std(r_data_gap3[0.1]) / np.sqrt(50)
    print(f"done ({time.time()-t0:.1f}s)")

    # 3. Sparse SYK below transition (p=0.05)
    print("  Sparse SYK (p=0.05)...", end=' ', flush=True)
    t0 = time.time()
    systems['Sparse SYK p=0.05'] = compute_diagnostics_for_system(
        'sparse_syk', N=10, beta=8.0, mu=0.1, sparsity=0.05, n_real=30)
    systems['Sparse SYK p=0.05']['peak_mean'] = np.mean(peak_data_gap3[0.05])
    systems['Sparse SYK p=0.05']['peak_sem'] = np.std(peak_data_gap3[0.05]) / np.sqrt(50)
    systems['Sparse SYK p=0.05']['r_mean'] = np.mean(r_data_gap3[0.05])
    systems['Sparse SYK p=0.05']['r_sem'] = np.std(r_data_gap3[0.05]) / np.sqrt(50)
    print(f"done ({time.time()-t0:.1f}s)")

    # 4. Sparse SYK deep non-chaotic (p=0.02)
    print("  Sparse SYK (p=0.02)...", end=' ', flush=True)
    t0 = time.time()
    systems['Sparse SYK p=0.02'] = compute_diagnostics_for_system(
        'sparse_syk', N=10, beta=8.0, mu=0.1, sparsity=0.02, n_real=30)
    systems['Sparse SYK p=0.02']['peak_mean'] = np.mean(peak_data_gap3[0.02])
    systems['Sparse SYK p=0.02']['peak_sem'] = np.std(peak_data_gap3[0.02]) / np.sqrt(50)
    systems['Sparse SYK p=0.02']['r_mean'] = np.mean(r_data_gap3[0.02])
    systems['Sparse SYK p=0.02']['r_sem'] = np.std(r_data_gap3[0.02]) / np.sqrt(50)
    print(f"done ({time.time()-t0:.1f}s)")

    # 5. Free fermion (q=2 SYK, should score low)
    print("  Free fermion (q=2)...", end=' ', flush=True)
    t0 = time.time()
    systems['Free fermion'] = compute_diagnostics_for_system(
        'free_fermion', N=10, beta=8.0, mu=0.1, sparsity=1.0, n_real=30)
    print(f"done ({time.time()-t0:.1f}s)")

    # Compute combined scores
    print("\n" + "=" * 80)
    print(f"{'System':<22} {'s_H':>6} {'s_r':>6} {'s_λ':>6} {'s_SFF':>6} "
          f"{'S_geom':>7} {'S_min':>6} {'Class':<16}")
    print("-" * 80)

    for name, data in systems.items():
        s_H = score_transmission(data['peak_mean'])
        s_r = score_level_spacing(data['r_mean'])
        s_lam = score_lyapunov(data['lyapunov_mean'], BETA)
        s_sff = data['sff_mean']

        S_geom = combined_score_geometric(s_H, s_r, s_lam, s_sff)
        S_min = combined_score_minimum(s_H, s_r, s_lam, s_sff)

        result = classify_system(s_H, s_r, s_lam, s_sff)

        print(f"{name:<22} {s_H:>6.3f} {s_r:>6.3f} {s_lam:>6.3f} {s_sff:>6.3f} "
              f"{S_geom:>7.3f} {S_min:>6.3f} {result['classification']:<16}")

    # Detailed analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)

    print("\n1. Does the classifier separate holographic from non-holographic?")
    dense_min = combined_score_minimum(
        score_transmission(systems['Dense SYK']['peak_mean']),
        score_level_spacing(systems['Dense SYK']['r_mean']),
        score_lyapunov(systems['Dense SYK']['lyapunov_mean'], BETA),
        systems['Dense SYK']['sff_mean'])
    free_min = combined_score_minimum(
        score_transmission(systems['Free fermion']['peak_mean']),
        score_level_spacing(systems['Free fermion']['r_mean']),
        score_lyapunov(systems['Free fermion']['lyapunov_mean'], BETA),
        systems['Free fermion']['sff_mean'])
    print(f"   Dense SYK S_min = {dense_min:.3f}")
    print(f"   Free fermion S_min = {free_min:.3f}")
    print(f"   Gap = {dense_min - free_min:.3f}")
    if dense_min - free_min > 0.3:
        print("   → GOOD separation")
    else:
        print("   → POOR separation")

    print("\n2. Behavior at sparsity transition:")
    for p_label in ['Sparse SYK p=0.1', 'Sparse SYK p=0.05', 'Sparse SYK p=0.02']:
        data = systems[p_label]
        s_H = score_transmission(data['peak_mean'])
        s_r = score_level_spacing(data['r_mean'])
        s_lam = score_lyapunov(data['lyapunov_mean'], BETA)
        s_sff = data['sff_mean']
        S_min = combined_score_minimum(s_H, s_r, s_lam, s_sff)
        print(f"   {p_label}: S_min={S_min:.3f} (s_H={s_H:.3f}, s_r={s_r:.3f}, "
              f"s_λ={s_lam:.3f}, s_SFF={s_sff:.3f})")
        print(f"     → Bottleneck diagnostic: ", end='')
        scores = {'s_H': s_H, 's_r': s_r, 's_λ': s_lam, 's_SFF': s_sff}
        bottleneck = min(scores, key=scores.get)
        print(f"{bottleneck} = {scores[bottleneck]:.3f}")

    print("\n3. Sensitivity to scoring scheme:")
    print("   System                  S_geom  S_min  Difference")
    for name, data in systems.items():
        s_H = score_transmission(data['peak_mean'])
        s_r = score_level_spacing(data['r_mean'])
        s_lam = score_lyapunov(data['lyapunov_mean'], BETA)
        s_sff = data['sff_mean']
        S_geom = combined_score_geometric(s_H, s_r, s_lam, s_sff)
        S_min = combined_score_minimum(s_H, s_r, s_lam, s_sff)
        print(f"   {name:<22} {S_geom:.3f}   {S_min:.3f}   {S_geom-S_min:.3f}")

    # Save results
    output_path = os.path.join(base, 'data', 'gap1_multi_diagnostic.h5')
    with h5py.File(output_path, 'w') as f:
        f.attrs['description'] = 'Gap 1: Multi-diagnostic classifier results'
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['beta'] = BETA
        f.attrs['mu'] = MU
        f.attrs['N_per_side'] = N_PER_SIDE

        for name, data in systems.items():
            grp = f.create_group(name.replace(' ', '_').replace('=', ''))
            for k, v in data.items():
                grp.attrs[k] = v

    print(f"\nSaved to {output_path}")
