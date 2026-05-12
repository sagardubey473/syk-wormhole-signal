"""Generate Gap 3 publication-quality plots."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Level spacing data
with h5py.File(os.path.join(base, 'data', 'gap3_level_spacing.h5'), 'r') as f:
    N_values = list(f.attrs['N_values'])
    sparsity_ls = list(f.attrs['sparsity_values'])

    r_data = {}
    for N in N_values:
        for p in sparsity_ls:
            key = f'N{N}_p{p:.3f}_r_means'
            r_data[(N, p)] = np.array(f[key])

# Transmission data
with h5py.File(os.path.join(base, 'data', 'gap3_transmission.h5'), 'r') as f:
    sparsity_tr = list(f.attrs['sparsity_values'])
    peak_data = {}
    for p in sparsity_tr:
        peak_data[p] = np.array(f[f'p_{p:.3f}']['peak_heights'])

results_dir = os.path.join(base, 'results')
os.makedirs(results_dir, exist_ok=True)

# === Figure 1: Level spacing ratio vs sparsity (N=10, N=14) ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, N in enumerate(N_values):
    ax = axes[ax_idx]
    means = [np.mean(r_data[(N, p)]) for p in sparsity_ls]
    sems = [np.std(r_data[(N, p)]) / np.sqrt(len(r_data[(N, p)])) for p in sparsity_ls]

    ax.errorbar(sparsity_ls, means, yerr=sems, fmt='o-', capsize=4, linewidth=2,
                markersize=8, color='C0', label=f'N={N} (50 real.)')
    ax.axhline(0.6027, color='red', linestyle='--', alpha=0.7, label='GUE (0.603)')
    ax.axhline(0.5307, color='orange', linestyle='--', alpha=0.7, label='GOE (0.531)')
    ax.axhline(0.3863, color='gray', linestyle='--', alpha=0.7, label='Poisson (0.386)')

    ax.set_xscale('log')
    ax.set_xlabel('Sparsity p', fontsize=13)
    ax.set_ylabel(r'$\langle r \rangle$', fontsize=13)
    ax.set_title(f'N={N} (N mod 8 = {N % 8}, GUE class)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.15, 0.7)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '07_gap3_level_spacing.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_gap3_level_spacing.png")

# === Figure 2: Transmission peak vs sparsity with error bars ===
fig, ax = plt.subplots(figsize=(8, 5))

means = [np.mean(peak_data[p]) for p in sparsity_tr]
sems = [np.std(peak_data[p]) / np.sqrt(len(peak_data[p])) for p in sparsity_tr]
stds = [np.std(peak_data[p]) for p in sparsity_tr]

ax.errorbar(sparsity_tr, means, yerr=sems, fmt='s-', capsize=4, linewidth=2,
            markersize=8, color='C1', label='Mean (SEM)')
ax.fill_between(sparsity_tr,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.2, color='C1', label=r'$\pm 1\sigma$ (std)')

ax.axhline(means[0], color='gray', linestyle=':', alpha=0.5,
           label=f'Dense mean = {means[0]:.4f}')
ax.set_xscale('log')
ax.set_xlabel('Sparsity p', fontsize=13)
ax.set_ylabel('Peak |C(t*)|', fontsize=13)
ax.set_title(f'Transmission Peak vs Sparsity (N=10, beta=8, mu=0.1, 50 realizations)', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0.7, 1.05)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '07_gap3_transmission_peak.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_gap3_transmission_peak.png")

# === Figure 3: Combined - signal vs chaos diagnostic (parametric plot) ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: both on same x-axis
ax = axes[0]
p_common = [p for p in sparsity_tr if p in sparsity_ls]

peak_means = [np.mean(peak_data[p]) for p in p_common]
peak_sems = [np.std(peak_data[p]) / np.sqrt(len(peak_data[p])) for p in p_common]
r_means_10 = [np.mean(r_data[(10, p)]) for p in p_common]
r_sems_10 = [np.std(r_data[(10, p)]) / np.sqrt(len(r_data[(10, p)])) for p in p_common]

ax2 = ax.twinx()
ln1 = ax.errorbar(p_common, peak_means, yerr=peak_sems, fmt='s-', capsize=3,
                   color='C1', linewidth=2, markersize=7, label='Peak |C|')
ln2 = ax2.errorbar(p_common, r_means_10, yerr=r_sems_10, fmt='o--', capsize=3,
                    color='C0', linewidth=2, markersize=7, label=r'$\langle r \rangle$ (N=10)')
ax2.axhline(0.6027, color='C0', linestyle=':', alpha=0.4)
ax2.axhline(0.3863, color='gray', linestyle=':', alpha=0.4)

ax.set_xscale('log')
ax.set_xlabel('Sparsity p', fontsize=13)
ax.set_ylabel('Peak |C(t*)|', fontsize=13, color='C1')
ax2.set_ylabel(r'$\langle r \rangle$', fontsize=13, color='C0')
ax.set_ylim(0.85, 0.96)
ax2.set_ylim(0.15, 0.7)
ax.invert_xaxis()

lines = [ln1, ln2]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=10, loc='lower left')
ax.set_title('Signal-Chaos Decoupling (N=10, 50 real.)', fontsize=12)
ax.grid(True, alpha=0.3)

# Right panel: parametric plot (r vs peak)
ax = axes[1]
ax.errorbar(r_means_10, peak_means,
            xerr=r_sems_10, yerr=peak_sems,
            fmt='D', capsize=4, markersize=8, color='C2', linewidth=2)

for i, p in enumerate(p_common):
    ax.annotate(f'p={p}', (r_means_10[i], peak_means[i]),
                textcoords="offset points", xytext=(8, -12 if i % 2 == 0 else 8),
                fontsize=8)

ax.axvline(0.6027, color='red', linestyle='--', alpha=0.5, label='GUE')
ax.axvline(0.3863, color='gray', linestyle='--', alpha=0.5, label='Poisson')
ax.set_xlabel(r'$\langle r \rangle$ (N=10)', fontsize=13)
ax.set_ylabel('Peak |C(t*)|', fontsize=13)
ax.set_title('Parametric: Peak vs Level Spacing', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0.85, 0.96)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '07_gap3_combined_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_gap3_combined_analysis.png")

# === Figure 4: Disorder fluctuations vs sparsity ===
fig, ax = plt.subplots(figsize=(8, 5))
stds_list = [np.std(peak_data[p]) for p in sparsity_tr]
ax.plot(sparsity_tr, stds_list, 'o-', linewidth=2, markersize=8, color='C3')
ax.set_xscale('log')
ax.set_xlabel('Sparsity p', fontsize=13)
ax.set_ylabel('Std dev of peak height', fontsize=13)
ax.set_title('Disorder Fluctuations Grow with Sparsification', fontsize=12)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '07_gap3_disorder_fluctuations.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_gap3_disorder_fluctuations.png")

print("\nAll Gap 3 plots generated.")
