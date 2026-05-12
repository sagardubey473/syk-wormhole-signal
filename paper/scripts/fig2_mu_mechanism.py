"""
Figure 2: Coupling mu controls the transmitted signal, not sparsity.

Peak height |C(t*)| vs mu for three sparsity levels (p = 1.0, 0.1, 0.05).
The three curves overlap, demonstrating that the coupling strength mu is the
relevant control parameter, independent of whether the internal dynamics are
chaotic, edge-of-chaos, or non-chaotic.
"""

import sys
import os
import numpy as np
import h5py

# Add scripts directory so we can import plot_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import setup_style, ensure_fig_dir, FIG_DIR, DATA_DIR, SINGLE_COL_WIDTH, COLORS

import matplotlib.pyplot as plt


def main():
    setup_style()
    # Guard against incomplete LaTeX installations: if usetex was enabled
    # but required packages (e.g. type1cm.sty) are missing, fall back to
    # mathtext so the script can still produce the figure.
    if plt.rcParams.get('text.usetex', False):
        try:
            fig_test, ax_test = plt.subplots()
            ax_test.set_xlabel(r'$\mu$')
            fig_test.savefig(os.devnull, format='pdf')
            plt.close(fig_test)
        except Exception:
            plt.rcParams.update({'text.usetex': False, 'mathtext.fontset': 'cm'})
    ensure_fig_dir()

    # ── Load data ──────────────────────────────────────────────────────
    data_path = os.path.join(DATA_DIR, 'research', 'mu_mechanism.h5')
    with h5py.File(data_path, 'r') as f:
        sparsity_values = f.attrs['sparsity_values']   # [1.0, 0.1, 0.05]
        mu_values = f.attrs['mu_values']                # 8 values
        peak_heights = f['peak_heights'][:]             # (3, 8, 50)

    # ── Compute mean and SEM over realizations (axis 2) ────────────────
    means = np.mean(peak_heights, axis=2)               # (3, 8)
    sems = np.std(peak_heights, axis=2, ddof=1) / np.sqrt(peak_heights.shape[2])  # (3, 8)

    # ── Curve styling ──────────────────────────────────────────────────
    curve_specs = [
        {'idx': 0, 'color': COLORS['blue'],   'marker': 'o', 'label': r'$p = 1.0$ (chaotic)'},
        {'idx': 1, 'color': COLORS['orange'], 'marker': 's', 'label': r'$p = 0.1$ (edge)'},
        {'idx': 2, 'color': COLORS['green'],  'marker': '^', 'label': r'$p = 0.05$ (non-chaotic)'},
    ]

    # ── Create figure ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.5))

    for spec in curve_specs:
        i = spec['idx']
        ax.errorbar(
            mu_values, means[i], yerr=sems[i],
            color=spec['color'],
            marker=spec['marker'],
            label=spec['label'],
            capsize=2,
            capthick=0.8,
            markeredgecolor=spec['color'],
            markerfacecolor=spec['color'],
            markersize=5,
            linewidth=1.2,
        )

    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'Peak $|C(t^*)|$')
    ax.legend()

    fig.tight_layout()

    # ── Save ───────────────────────────────────────────────────────────
    out_path = os.path.join(FIG_DIR, 'fig2_mu_mechanism.pdf')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
