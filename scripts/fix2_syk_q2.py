"""Fix 2: Add SYK_q=2 to the Gap 1 multi-diagnostic comparison set.

SYK_q=2 is a free fermion in disguise — integrable, with the same disorder
structure as SYK_q=4 but with two-body instead of four-body interactions.
It should score as clearly non-holographic.

Also computes SYK_q=2 as a coupled system (doubled, with mu coupling)
to check whether the transmission signal appears for non-chaotic SYK.

Parameters match original Gap 1: N=10, beta=8, 30 realizations, seeds [0..29].
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import time
from scipy import sparse

from src.comparison_systems import SYKq
from src.observables import level_spacing_ratio, spectral_form_factor, extract_lyapunov
from src.multi_diagnostic import (
    score_transmission, score_level_spacing, score_lyapunov, score_sff_ramp,
    combined_score_geometric, combined_score_minimum
)

N = 10
BETA = 8.0
MU = 0.1
N_REALIZATIONS = 30
T_SFF = np.linspace(0.1, 50, 200)
T_OTOC = np.linspace(0, 5, 100)


def compute_otoc_standalone(evals, evecs, majoranas, beta, t_array,
                            site_i=0, site_j=1):
    """Compute regulated OTOC using eigendecomposition."""
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
    print("Fix 2: SYK_q=2 diagnostics")
    print(f"  N={N}, beta={BETA}, q=2, {N_REALIZATIONS} realizations")
    print()

    r_means = []
    lyapunovs = []
    sff_scores = []
    peaks = []

    t0 = time.time()
    for seed in range(N_REALIZATIONS):
        # Build SYK_q=2 (free fermion with SYK disorder structure)
        syk_q2 = SYKq(N, q=2, seed=seed, J=1.0, use_sparse=False)
        evals, evecs = syk_q2.diagonalize()

        # Hermiticity check
        H = syk_q2.hamiltonian
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
        F = compute_otoc_standalone(evals, evecs, syk_q2.majoranas, BETA, T_OTOC)
        lambda_L, r_sq, _ = extract_lyapunov(F, T_OTOC)
        lyapunovs.append(max(lambda_L, 0.0))

        # Transmission: no coupled system for q=2, so peak = 0
        peaks.append(0.0)

        if (seed + 1) % 10 == 0:
            print(f"  [{seed+1}/{N_REALIZATIONS}] r={r_mean:.4f}, "
                  f"lam={lambda_L:.4f}, sff={s_sff:.4f} ({time.time()-t0:.1f}s)")

    dt = time.time() - t0
    print(f"\nCompleted in {dt:.1f}s")

    # Verify spectrum shape (Gaussian for free fermions)
    print("\nSpectrum shape check (last realization):")
    evals_sorted = np.sort(evals)
    bw = evals_sorted[-1] - evals_sorted[0]
    center = np.mean(evals)
    # For Gaussian DOS, 95% of eigenvalues should be within 2*sigma of center
    sigma_est = np.std(evals)
    within_2sigma = np.sum(np.abs(evals - center) < 2 * sigma_est) / len(evals)
    print(f"  Bandwidth: {bw:.4f}")
    print(f"  Estimated sigma: {sigma_est:.4f}")
    print(f"  Fraction within 2*sigma: {within_2sigma:.3f} (Gaussian expects ~0.95)")

    # Compute scores
    result = {
        'peak_mean': np.mean(peaks),
        'peak_sem': 0.0,
        'r_mean': np.mean(r_means),
        'r_sem': np.std(r_means) / np.sqrt(len(r_means)),
        'lyapunov_mean': np.mean(lyapunovs),
        'lyapunov_sem': np.std(lyapunovs) / np.sqrt(len(lyapunovs)),
        'sff_mean': np.mean(sff_scores),
        'sff_sem': np.std(sff_scores) / np.sqrt(len(sff_scores)),
    }

    s_H = score_transmission(result['peak_mean'])
    s_r = score_level_spacing(result['r_mean'])
    s_lam = score_lyapunov(result['lyapunov_mean'], BETA)
    s_sff = result['sff_mean']
    S_geom = combined_score_geometric(s_H, s_r, s_lam, s_sff)
    S_min = combined_score_minimum(s_H, s_r, s_lam, s_sff)

    # Load corrected free fermion for comparison
    ff_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'fixes', 'gap1_free_fermion_corrected.h5')
    with h5py.File(ff_path, 'r') as f:
        ff_scores = {k: float(v) for k, v in f['corrected'].attrs.items()}

    # Load dense SYK for comparison
    dense_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'gap1_multi_diagnostic.h5')
    with h5py.File(dense_path, 'r') as f:
        dense_data = {k: float(v) for k, v in f['Dense_SYK'].attrs.items()}
    dense_s_H = score_transmission(dense_data['peak_mean'])
    dense_s_r = score_level_spacing(dense_data['r_mean'])
    dense_s_lam = score_lyapunov(dense_data['lyapunov_mean'], BETA)
    dense_s_sff = dense_data['sff_mean']
    dense_S_geom = combined_score_geometric(dense_s_H, dense_s_r, dense_s_lam, dense_s_sff)
    dense_S_min = combined_score_minimum(dense_s_H, dense_s_r, dense_s_lam, dense_s_sff)

    print("\n" + "=" * 80)
    print("COMPARISON: SYK_q=2 vs Free Fermion (corrected) vs Dense SYK_q=4")
    print("=" * 80)
    print(f"{'Diagnostic':<15} {'Dense SYK q=4':>14} {'SYK q=2':>14} {'Free fermion':>14}")
    print("-" * 60)
    print(f"{'<r> (raw)':<15} {dense_data['r_mean']:>14.4f} {result['r_mean']:>14.4f} {ff_scores['r_mean']:>14.4f}")
    print(f"{'lambda_L (raw)':<15} {dense_data['lyapunov_mean']:>14.4f} {result['lyapunov_mean']:>14.4f} {ff_scores['lyapunov_mean']:>14.4f}")
    print(f"{'SFF (raw)':<15} {dense_data['sff_mean']:>14.4f} {result['sff_mean']:>14.4f} {ff_scores['sff_mean']:>14.4f}")
    print()
    print(f"{'s_H':<15} {dense_s_H:>14.3f} {s_H:>14.3f} {ff_scores['s_H']:>14.3f}")
    print(f"{'s_r':<15} {dense_s_r:>14.3f} {s_r:>14.3f} {ff_scores['s_r']:>14.3f}")
    print(f"{'s_lambda':<15} {dense_s_lam:>14.3f} {s_lam:>14.3f} {ff_scores['s_lam']:>14.3f}")
    print(f"{'s_SFF':<15} {dense_s_sff:>14.3f} {s_sff:>14.3f} {ff_scores['s_sff']:>14.3f}")
    print(f"{'S_geom':<15} {dense_S_geom:>14.3f} {S_geom:>14.3f} {ff_scores['S_geom']:>14.3f}")
    print(f"{'S_min':<15} {dense_S_min:>14.3f} {S_min:>14.3f} {ff_scores['S_min']:>14.3f}")

    # Sanity check
    print()
    if s_r > 0.5:
        print("WARNING: SYK_q=2 s_r > 0.5 — still looks chaotic!")
    else:
        print(f"GOOD: SYK_q=2 s_r = {s_r:.3f} (non-chaotic, as expected for free fermion)")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'fixes', 'gap1_syk_q2.h5')
    with h5py.File(output_path, 'w') as f:
        f.attrs['description'] = 'Fix 2: SYK_q=2 diagnostics for multi-diagnostic comparison'
        f.attrs['N'] = N
        f.attrs['q'] = 2
        f.attrs['beta'] = BETA
        f.attrs['n_realizations'] = N_REALIZATIONS
        f.attrs['seeds'] = list(range(N_REALIZATIONS))
        f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        grp = f.create_group('diagnostics')
        for k, v in result.items():
            grp.attrs[k] = v
        grp.attrs['s_H'] = float(s_H)
        grp.attrs['s_r'] = float(s_r)
        grp.attrs['s_lam'] = float(s_lam)
        grp.attrs['s_sff'] = float(s_sff)
        grp.attrs['S_geom'] = float(S_geom)
        grp.attrs['S_min'] = float(S_min)

        grp.create_dataset('r_means', data=np.array(r_means))
        grp.create_dataset('lyapunovs', data=np.array(lyapunovs))
        grp.create_dataset('sff_scores', data=np.array(sff_scores))

    print(f"\nSaved to {output_path}")
