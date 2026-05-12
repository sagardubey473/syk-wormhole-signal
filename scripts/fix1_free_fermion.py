"""Fix 1: Rerun free fermion diagnostics with correct Hamiltonian.

The original Gap 1 script always used SYKHamiltonian (q=4) for spectral
diagnostics regardless of system_type. This fix uses FreeFermionSystem
(quadratic Hamiltonian, integrable) for the free fermion entry.

Parameters match original Gap 1: N=10, beta=8, 30 realizations, seeds [0..29].
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import time
from scipy import sparse

from src.comparison_systems import FreeFermionSystem
from src.observables import level_spacing_ratio, spectral_form_factor, extract_lyapunov
from src.multi_diagnostic import (
    score_transmission, score_level_spacing, score_lyapunov, score_sff_ramp,
    combined_score_geometric, combined_score_minimum
)

N = 10
BETA = 8.0
N_REALIZATIONS = 30
T_SFF = np.linspace(0.1, 50, 200)
T_OTOC = np.linspace(0, 5, 100)


def compute_otoc_standalone(evals, evecs, majoranas, beta, t_array,
                            site_i=0, site_j=1):
    """Compute regulated OTOC using eigendecomposition.

    F(t) = Tr[rho^{1/4} W(t) rho^{1/4} V rho^{1/4} W(t) rho^{1/4} V]
    """
    dim = len(evals)
    rho_quarter_eig = np.exp(-beta * evals / 4.0)
    rho_quarter_eig /= np.sum(np.exp(-beta * evals)) ** 0.25

    W = majoranas[site_i]
    V = majoranas[site_j]
    if sparse.issparse(W):
        W = W.toarray()
    if sparse.issparse(V):
        V = V.toarray()
    W_eig = evecs.conj().T @ W @ evecs
    V_eig = evecs.conj().T @ V @ evecs
    rho_q = np.diag(rho_quarter_eig)

    F = np.zeros(len(t_array), dtype=complex)
    for idx, t in enumerate(t_array):
        phases = np.exp(1j * evals * t)
        Wt_eig = np.diag(phases) @ W_eig @ np.diag(phases.conj())
        A = rho_q @ Wt_eig @ rho_q @ V_eig @ rho_q @ Wt_eig @ rho_q @ V_eig
        F[idx] = np.trace(A)

    return F


if __name__ == '__main__':
    print("Fix 1: Corrected free fermion diagnostics")
    print(f"  N={N}, beta={BETA}, {N_REALIZATIONS} realizations")
    print()

    r_means = []
    lyapunovs = []
    sff_scores = []

    t0 = time.time()
    for seed in range(N_REALIZATIONS):
        # Use FreeFermionSystem (quadratic, integrable)
        ff = FreeFermionSystem(N, seed=seed, J=1.0, use_sparse=False)
        evals, evecs = ff.diagonalize()

        # Hermiticity check
        H = ff.hamiltonian
        herm_err = np.linalg.norm(H - H.conj().T)
        assert herm_err < 1e-10, f"Hermiticity failed: err={herm_err}"

        # Level spacing
        _, r_mean = level_spacing_ratio(evals)
        r_means.append(r_mean)

        # SFF
        dim_single = 2 ** (N // 2)
        K = spectral_form_factor(evals, T_SFF)
        s_sff = score_sff_ramp(K, T_SFF, dim_single)
        sff_scores.append(s_sff)

        # OTOC and Lyapunov
        F = compute_otoc_standalone(evals, evecs, ff.majoranas, BETA, T_OTOC)
        lambda_L, r_sq, _ = extract_lyapunov(F, T_OTOC)
        lyapunovs.append(max(lambda_L, 0.0))

        if (seed + 1) % 10 == 0:
            print(f"  [{seed+1}/{N_REALIZATIONS}] r={r_mean:.4f}, "
                  f"lam={lambda_L:.4f}, sff={s_sff:.4f} ({time.time()-t0:.1f}s)")

    dt = time.time() - t0
    print(f"\nCompleted in {dt:.1f}s")

    # Compute corrected scores
    corrected = {
        'peak_mean': 0.0,  # Free fermion has no wormhole signal (unchanged)
        'peak_sem': 0.0,
        'r_mean': np.mean(r_means),
        'r_sem': np.std(r_means) / np.sqrt(len(r_means)),
        'lyapunov_mean': np.mean(lyapunovs),
        'lyapunov_sem': np.std(lyapunovs) / np.sqrt(len(lyapunovs)),
        'sff_mean': np.mean(sff_scores),
        'sff_sem': np.std(sff_scores) / np.sqrt(len(sff_scores)),
    }

    s_H = score_transmission(corrected['peak_mean'])
    s_r = score_level_spacing(corrected['r_mean'])
    s_lam = score_lyapunov(corrected['lyapunov_mean'], BETA)
    s_sff = corrected['sff_mean']
    S_geom = combined_score_geometric(s_H, s_r, s_lam, s_sff)
    S_min = combined_score_minimum(s_H, s_r, s_lam, s_sff)

    # Original buggy values (from gap1_multi_diagnostic.h5)
    buggy = {
        'peak_mean': 0.0, 'r_mean': 0.5959,
        'lyapunov_mean': 0.5759, 'sff_mean': 0.9583,
        's_H': 0.000, 's_r': 0.969, 's_lam': 0.733, 's_sff': 0.958,
        'S_geom': 0.162, 'S_min': 0.000,
    }

    print("\n" + "=" * 70)
    print("FREE FERMION DIAGNOSTIC COMPARISON")
    print("=" * 70)
    print(f"{'Diagnostic':<20} {'Buggy (q=4 SYK)':>16} {'Corrected (FF)':>16} {'Change':>10}")
    print("-" * 65)
    print(f"{'<r> (raw)':.<20} {buggy['r_mean']:>16.4f} {corrected['r_mean']:>16.4f} "
          f"{'FIXED' if abs(corrected['r_mean'] - buggy['r_mean']) > 0.05 else 'same':>10}")
    print(f"{'lambda_L (raw)':.<20} {buggy['lyapunov_mean']:>16.4f} {corrected['lyapunov_mean']:>16.4f} "
          f"{'FIXED' if abs(corrected['lyapunov_mean'] - buggy['lyapunov_mean']) > 0.05 else 'same':>10}")
    print(f"{'SFF score (raw)':.<20} {buggy['sff_mean']:>16.4f} {corrected['sff_mean']:>16.4f} "
          f"{'FIXED' if abs(corrected['sff_mean'] - buggy['sff_mean']) > 0.05 else 'same':>10}")
    print()
    print(f"{'s_H':.<20} {buggy['s_H']:>16.3f} {s_H:>16.3f}")
    print(f"{'s_r':.<20} {buggy['s_r']:>16.3f} {s_r:>16.3f}")
    print(f"{'s_lambda':.<20} {buggy['s_lam']:>16.3f} {s_lam:>16.3f}")
    print(f"{'s_SFF':.<20} {buggy['s_sff']:>16.3f} {s_sff:>16.3f}")
    print(f"{'S_geom':.<20} {buggy['S_geom']:>16.3f} {S_geom:>16.3f}")
    print(f"{'S_min':.<20} {buggy['S_min']:>16.3f} {S_min:>16.3f}")
    print()

    # Sanity check
    if s_r > 0.5:
        print("WARNING: Free fermion s_r > 0.5 — still looks chaotic! Fix may be incomplete.")
    else:
        print(f"GOOD: Free fermion s_r = {s_r:.3f} (non-chaotic, as expected)")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'fixes', 'gap1_free_fermion_corrected.h5')
    with h5py.File(output_path, 'w') as f:
        f.attrs['description'] = 'Fix 1: Corrected free fermion diagnostics using FreeFermionSystem'
        f.attrs['N'] = N
        f.attrs['beta'] = BETA
        f.attrs['n_realizations'] = N_REALIZATIONS
        f.attrs['seeds'] = list(range(N_REALIZATIONS))
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        grp_corrected = f.create_group('corrected')
        for k, v in corrected.items():
            grp_corrected.attrs[k] = v
        grp_corrected.attrs['s_H'] = float(s_H)
        grp_corrected.attrs['s_r'] = float(s_r)
        grp_corrected.attrs['s_lam'] = float(s_lam)
        grp_corrected.attrs['s_sff'] = float(s_sff)
        grp_corrected.attrs['S_geom'] = float(S_geom)
        grp_corrected.attrs['S_min'] = float(S_min)

        grp_corrected.create_dataset('r_means', data=np.array(r_means))
        grp_corrected.create_dataset('lyapunovs', data=np.array(lyapunovs))
        grp_corrected.create_dataset('sff_scores', data=np.array(sff_scores))

        grp_buggy = f.create_group('original_buggy')
        for k, v in buggy.items():
            grp_buggy.attrs[k] = v

    print(f"\nSaved to {output_path}")
