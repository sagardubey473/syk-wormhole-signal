#!/usr/bin/env python3
"""
Figure 3: Noise-sparsity factorization.

Panel (a): Peak height vs dephasing rate gamma for each sparsity level.
           Curves overlap, demonstrating sparsity-independent noise sensitivity.
Panel (b): Critical noise gamma* (50% degradation threshold) vs sparsity.

Data: gap2_noise_sparsity.h5
"""

import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator

# Add scripts directory for plot_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import (setup_style, ensure_fig_dir, FIG_DIR, DATA_DIR,
                        DOUBLE_COL_WIDTH, COLORS, SPARSITY_COLORS,
                        SPARSITY_MARKERS)


def load_data():
    """Load peak heights from the noise-sparsity HDF5 file."""
    h5_path = os.path.join(DATA_DIR, 'gap2_noise_sparsity.h5')
    data = {}  # (p, gamma) -> array of peak heights

    with h5py.File(h5_path, 'r') as f:
        sparsity_values = f.attrs['sparsity_values']
        gamma_values = f.attrs['gamma_values']

        for p in sparsity_values:
            for gamma in gamma_values:
                key = f'p_{p:.3f}/gamma_{gamma:.4f}/peak_heights'
                data[(p, gamma)] = f[key][:]

    return sparsity_values, gamma_values, data


def compute_gamma_star(sparsity_values, gamma_values, data):
    """
    Compute gamma* (50% degradation threshold) for each sparsity.

    For each sparsity p, find the gamma where the mean peak height crosses
    0.5 * mean_peak_at_gamma_0 by linear interpolation in log-gamma space
    between the two surrounding gamma values.
    """
    gamma_star = {}
    for p in sparsity_values:
        # Mean peak at gamma=0
        clean_mean = np.mean(data[(p, 0.0)])
        threshold = 0.5 * clean_mean

        # Compute mean peaks for all gamma > 0
        gammas_nonzero = [g for g in gamma_values if g > 0]
        means = [np.mean(data[(p, g)]) for g in gamma_values]

        # Find crossing: walk through all gamma values (including 0)
        # and find where mean crosses the threshold
        for i in range(len(gamma_values) - 1):
            m_lo = means[i]
            m_hi = means[i + 1]
            g_lo = gamma_values[i]
            g_hi = gamma_values[i + 1]

            if (m_lo >= threshold) and (m_hi < threshold):
                # Linear interpolation in log-gamma space
                # Handle g_lo = 0 by using a small placeholder for log
                if g_lo == 0:
                    # Use the actual gamma=0 mean but interpolate starting
                    # from the next nonzero gamma boundary
                    # Since gamma=0 is exact, use log of a small value
                    log_lo = np.log10(5e-4)
                else:
                    log_lo = np.log10(g_lo)
                log_hi = np.log10(g_hi)

                # Fraction of the way from lo to hi where threshold is crossed
                frac = (m_lo - threshold) / (m_lo - m_hi)
                log_gstar = log_lo + frac * (log_hi - log_lo)
                gamma_star[p] = 10**log_gstar
                break

    return gamma_star


def main():
    setup_style()
    # Ensure fallback if LaTeX packages are incomplete
    try:
        fig_test = plt.figure()
        fig_test.text(0.5, 0.5, r'$\gamma$')
        fig_test.savefig(os.devnull, format='pdf')
        plt.close(fig_test)
    except Exception:
        plt.rcParams.update({'text.usetex': False, 'mathtext.fontset': 'cm'})
    ensure_fig_dir()

    sparsity_values, gamma_values, data = load_data()
    gamma_star = compute_gamma_star(sparsity_values, gamma_values, data)

    # --- Figure setup ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.8))
    fig.subplots_adjust(wspace=0.35)

    # Color and marker assignments for the four sparsity levels
    color_list = [SPARSITY_COLORS[p] for p in sparsity_values]
    marker_list = [SPARSITY_MARKERS[p] for p in sparsity_values]

    # ========== Panel (a): Peak height vs gamma ==========

    # Compute the overall clean mean (gamma=0) for the 50% threshold line
    # Use mean across all sparsities at gamma=0
    clean_means = [np.mean(data[(p, 0.0)]) for p in sparsity_values]
    overall_clean_mean = np.mean(clean_means)

    for idx, p in enumerate(sparsity_values):
        means = []
        sems = []
        x_vals = []
        for gamma in gamma_values:
            peaks = data[(p, gamma)]
            m = np.mean(peaks)
            sem = np.std(peaks, ddof=1) / np.sqrt(len(peaks))
            means.append(m)
            sems.append(sem)
            # Plot gamma=0 at x=5e-4
            if gamma == 0.0:
                x_vals.append(5e-4)
            else:
                x_vals.append(gamma)

        x_vals = np.array(x_vals)
        means = np.array(means)
        sems = np.array(sems)

        label = f'$p={p}$'
        ax_a.errorbar(x_vals, means, yerr=sems,
                      color=color_list[idx],
                      marker=marker_list[idx],
                      markersize=5,
                      capsize=2,
                      linewidth=1.2,
                      label=label)

    # 50% threshold line
    use_tex = plt.rcParams.get('text.usetex', False)
    thresh_label = r'50\% threshold' if use_tex else '50% threshold'
    ax_a.axhline(0.5 * overall_clean_mean, color=COLORS['gray'],
                 linestyle='--', linewidth=0.8, zorder=0,
                 label=thresh_label)

    ax_a.set_xscale('log')
    ax_a.set_xlabel(r'$\gamma_\phi$')
    ax_a.set_ylabel(r'Peak $|C(t^*)|$')
    ax_a.legend(fontsize=7, loc='best')

    # Custom tick for gamma=0 at 5e-4
    current_ticks = [5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    current_labels = ['0', r'$10^{-3}$', r'$3{\times}10^{-3}$',
                      r'$10^{-2}$', r'$3{\times}10^{-2}$', r'$10^{-1}$']
    ax_a.set_xticks(current_ticks)
    ax_a.set_xticklabels(current_labels, fontsize=7)
    ax_a.xaxis.set_minor_locator(NullLocator())
    ax_a.minorticks_off()

    ax_a.set_xlim(3e-4, 0.15)

    # Panel label
    ax_a.text(0.03, 0.95, r'\textbf{(a)}' if plt.rcParams.get('text.usetex', False)
              else r'$\mathbf{(a)}$',
              transform=ax_a.transAxes, fontsize=11,
              verticalalignment='top', fontweight='bold')

    # ========== Panel (b): gamma* vs sparsity ==========
    p_vals = sorted(gamma_star.keys(), reverse=True)
    gstar_vals = [gamma_star[p] for p in p_vals]

    ax_b.plot(p_vals, gstar_vals,
              marker='s', color=COLORS['blue'],
              markersize=7, linewidth=0,
              markeredgecolor='black', markeredgewidth=0.5)

    ax_b.set_xscale('log')
    ax_b.set_xlabel(r'Sparsity $p$')
    ax_b.set_ylabel(r'$\gamma^*$')
    ax_b.set_ylim(0.054, 0.057)

    # Panel label
    ax_b.text(0.03, 0.95, r'\textbf{(b)}' if plt.rcParams.get('text.usetex', False)
              else r'$\mathbf{(b)}$',
              transform=ax_b.transAxes, fontsize=11,
              verticalalignment='top', fontweight='bold')

    # --- Save ---
    out_path = os.path.join(FIG_DIR, 'fig3_noise.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print gamma* values for reference
    print("\ngamma* values:")
    for p in sorted(gamma_star.keys(), reverse=True):
        print(f"  p={p:.3f}: gamma* = {gamma_star[p]:.6f}")


if __name__ == '__main__':
    main()
