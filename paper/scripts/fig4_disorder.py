#!/usr/bin/env python3
"""
Figure 4: Disorder fluctuations grow while mean is preserved.

Violin plot of peak height distributions across 9 sparsity values,
showing that while the mean peak height stays roughly constant,
the variance increases substantially as sparsity decreases.
"""

import sys
import os
import numpy as np
import h5py

# Add scripts directory so we can import plot_style
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import setup_style, ensure_fig_dir, FIG_DIR, DATA_DIR, SINGLE_COL_WIDTH

import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup_style()
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
ensure_fig_dir()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
h5_path = os.path.join(DATA_DIR, 'gap3_transmission.h5')
with h5py.File(h5_path, 'r') as f:
    sparsity_values = list(f.attrs['sparsity_values'])  # [1.0, 0.5, ..., 0.02]
    peak_data = []
    for p in sparsity_values:
        key = f'p_{p:.3f}/peak_heights'
        peak_data.append(np.array(f[key]))

n_sparsities = len(sparsity_values)
positions = np.arange(n_sparsities)

# ---------------------------------------------------------------------------
# Color gradient: blue (dense, p=1.0) -> dark (sparse, p=0.02)
# ---------------------------------------------------------------------------
blue = np.array(mcolors.to_rgba('#0072B2'))
dark = np.array(mcolors.to_rgba('#1a1a1a'))
violin_colors = [mcolors.to_hex(blue + t * (dark - blue))
                 for t in np.linspace(0, 1, n_sparsities)]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.8))

# Violin plot
parts = ax.violinplot(peak_data, positions=positions,
                      showmeans=True, showmedians=False,
                      showextrema=False)

# Style the violin bodies
for i, body in enumerate(parts['bodies']):
    body.set_facecolor(violin_colors[i])
    body.set_edgecolor('black')
    body.set_linewidth(0.6)
    body.set_alpha(0.7)

# Style the mean lines
parts['cmeans'].set_color('black')
parts['cmeans'].set_linewidth(0.8)

# Overlay individual data points with jitter
rng = np.random.default_rng(42)
for i, data in enumerate(peak_data):
    jitter = rng.uniform(-0.15, 0.15, size=len(data))
    ax.scatter(positions[i] + jitter, data, s=4, alpha=0.4,
               color=violin_colors[i], edgecolors='none', zorder=3)

# ---------------------------------------------------------------------------
# Reference line: dense (p=1.0) mean
# ---------------------------------------------------------------------------
dense_mean = np.mean(peak_data[0])
ax.axhline(dense_mean, color='black', linestyle='--', linewidth=0.8,
           alpha=0.6, zorder=1)

# ---------------------------------------------------------------------------
# Annotate std at p=1.0 and p=0.02
# ---------------------------------------------------------------------------
std_dense = np.std(peak_data[0])
std_sparse = np.std(peak_data[-1])

ax.annotate(
    rf'$\sigma = {std_dense:.3f}$',
    xy=(0, dense_mean + std_dense),
    xytext=(1.5, np.max(peak_data[0]) + 0.015),
    fontsize=7,
    ha='center',
    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6),
    color='gray',
)

ax.annotate(
    rf'$\sigma = {std_sparse:.3f}$',
    xy=(n_sparsities - 1, np.mean(peak_data[-1]) + std_sparse),
    xytext=(n_sparsities - 2.5, np.min(peak_data[-1]) - 0.02),
    fontsize=7,
    ha='center',
    arrowprops=dict(arrowstyle='->', color='black', lw=0.6),
    color='black',
)

# ---------------------------------------------------------------------------
# Axes labels and ticks
# ---------------------------------------------------------------------------
ax.set_xticks(positions)
ax.set_xticklabels([f'{p}' for p in sparsity_values], rotation=45, ha='right')
ax.set_xlabel(r'Sparsity $p$')
ax.set_ylabel(r'Peak $|C(t^*)|$')

ax.set_xlim(-0.6, n_sparsities - 0.4)

# Minor y-ticks
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)

fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(FIG_DIR, 'fig4_disorder.pdf')
fig.savefig(out_path)
plt.close(fig)
print(f'Saved: {out_path}')
