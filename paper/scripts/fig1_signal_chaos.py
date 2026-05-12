#!/usr/bin/env python3
"""
Figure 1 -- Signal-chaos decoupling.

Two vertically stacked panels sharing a log-scale x-axis (sparsity p):
  (a) Level spacing ratio <r> vs sparsity p for N=10 and N=14.
  (b) Transmission peak height vs sparsity p.
"""

import sys
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Add scripts directory for plot_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import (setup_style, ensure_fig_dir,
                        FIG_DIR, DATA_DIR, DOUBLE_COL_WIDTH, COLORS)


def load_level_spacing(data_dir):
    """Load level spacing ratio data for N=10 and N=14."""
    fpath = os.path.join(data_dir, 'gap3_level_spacing.h5')
    with h5py.File(fpath, 'r') as f:
        sparsity_values = np.array(f.attrs['sparsity_values'])

        r_N10_means = []
        r_N10_sems = []
        r_N14_means = []
        r_N14_sems = []

        for p in sparsity_values:
            # N=10
            key10 = f'N10_p{p:.3f}_r_means'
            data10 = np.array(f[key10])
            r_N10_means.append(np.mean(data10))
            r_N10_sems.append(np.std(data10, ddof=1) / np.sqrt(len(data10)))

            # N=14
            key14 = f'N14_p{p:.3f}_r_means'
            data14 = np.array(f[key14])
            r_N14_means.append(np.mean(data14))
            r_N14_sems.append(np.std(data14, ddof=1) / np.sqrt(len(data14)))

    return (sparsity_values,
            np.array(r_N10_means), np.array(r_N10_sems),
            np.array(r_N14_means), np.array(r_N14_sems))


def load_transmission(data_dir):
    """Load transmission peak height data."""
    fpath = os.path.join(data_dir, 'gap3_transmission.h5')
    with h5py.File(fpath, 'r') as f:
        sparsity_values = np.array(f.attrs['sparsity_values'])

        peak_means = []
        peak_sems = []

        for p in sparsity_values:
            key = f'p_{p:.3f}'
            heights = np.array(f[key]['peak_heights'])
            peak_means.append(np.mean(heights))
            peak_sems.append(np.std(heights, ddof=1) / np.sqrt(len(heights)))

    return sparsity_values, np.array(peak_means), np.array(peak_sems)


def main():
    setup_style()
    # Force mathtext fallback if LaTeX packages are incomplete
    plt.rcParams.update({'text.usetex': False, 'mathtext.fontset': 'cm'})
    ensure_fig_dir()

    # Load data
    p_ls, r10_mean, r10_sem, r14_mean, r14_sem = load_level_spacing(DATA_DIR)
    p_tx, peak_mean, peak_sem = load_transmission(DATA_DIR)

    # Dense (p=1.0) reference value for transmission
    idx_dense = np.argmin(np.abs(p_tx - 1.0))
    dense_peak = peak_mean[idx_dense]

    # Reference values for level spacing
    GUE = 0.603
    POISSON = 0.386

    # Create figure: full width, two vertically stacked panels
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, sharex=True,
        figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.65),
        gridspec_kw={'hspace': 0.08}
    )

    # ── Panel (a): Level spacing ratio ──────────────────────────────
    ax_a.errorbar(
        p_ls, r10_mean, yerr=r10_sem,
        fmt='o', color=COLORS['blue'], markerfacecolor=COLORS['blue'],
        markersize=5, capsize=2, label=r'$N=10$', zorder=3
    )
    ax_a.errorbar(
        p_ls, r14_mean, yerr=r14_sem,
        fmt='s', color=COLORS['orange'], markerfacecolor='none',
        markeredgecolor=COLORS['orange'], markeredgewidth=1.2,
        markersize=5, capsize=2, label=r'$N=14$', zorder=3
    )

    # Reference lines
    ax_a.axhline(GUE, color='gray', linestyle='--', linewidth=0.8, zorder=1)
    ax_a.axhline(POISSON, color='gray', linestyle='--', linewidth=0.8,
                 zorder=1)

    # Labels for reference lines — place at right edge
    xlims = (min(p_ls) * 0.7, max(p_ls) * 1.5)
    ax_a.text(xlims[1] * 0.95, GUE + 0.008, 'GUE',
              ha='right', va='bottom', fontsize=8, color='gray')
    ax_a.text(xlims[1] * 0.95, POISSON - 0.008, 'Poisson',
              ha='right', va='top', fontsize=8, color='gray')

    ax_a.set_ylabel(r'$\langle r \rangle$')
    ax_a.set_xlim(xlims)
    ax_a.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                borderaxespad=0, handletextpad=0.4)
    ax_a.text(0.02, 0.95, '(a)', transform=ax_a.transAxes,
              fontsize=10, fontweight='bold', va='top', ha='left')

    # Remove x-tick labels for top panel (shared axis)
    plt.setp(ax_a.get_xticklabels(), visible=False)

    # ── Panel (b): Transmission peak height ─────────────────────────
    ax_b.errorbar(
        p_tx, peak_mean, yerr=peak_sem,
        fmt='o', color=COLORS['blue'], markerfacecolor=COLORS['blue'],
        markersize=5, capsize=2, zorder=3
    )

    # Dense reference line
    ax_b.axhline(dense_peak, color='gray', linestyle='--', linewidth=0.8,
                 zorder=1)

    ax_b.set_ylabel(r'Peak $|C(t^*)|$')
    ax_b.set_xlabel(r'Sparsity $p$')
    ax_b.set_xscale('log')
    ax_b.set_xlim(xlims)
    ax_b.text(0.02, 0.95, '(b)', transform=ax_b.transAxes,
              fontsize=10, fontweight='bold', va='top', ha='left')

    # Save
    outpath = os.path.join(FIG_DIR, 'fig1_signal_chaos.pdf')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()
