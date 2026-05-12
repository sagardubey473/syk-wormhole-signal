"""Generate Gap 2 publication-quality plots."""

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

with h5py.File(os.path.join(base, 'data', 'gap2_noise_sparsity.h5'), 'r') as f:
    sparsity_values = list(f.attrs['sparsity_values'])
    gamma_values = list(f.attrs['gamma_values'])

    peaks = {}
    for p in sparsity_values:
        for gamma in gamma_values:
            key = (p, gamma)
            grp = f[f'p_{p:.3f}'][f'gamma_{gamma:.4f}']
            peaks[key] = np.array(grp['peak_heights'])

results_dir = os.path.join(base, 'results')
os.makedirs(results_dir, exist_ok=True)

colors_p = {1.0: 'C0', 0.3: 'C1', 0.1: 'C2', 0.05: 'C3'}
markers_p = {1.0: 'o', 0.3: 's', 0.1: 'D', 0.05: '^'}

# === Figure 1: Peak height vs gamma for each sparsity ===
fig, ax = plt.subplots(figsize=(9, 6))

for p in sparsity_values:
    means = [np.mean(peaks[(p, g)]) for g in gamma_values]
    sems = [np.std(peaks[(p, g)]) / np.sqrt(len(peaks[(p, g)])) for g in gamma_values]

    # Offset gamma=0 slightly for log scale
    gamma_plot = [g if g > 0 else 5e-4 for g in gamma_values]

    ax.errorbar(gamma_plot, means, yerr=sems, fmt=f'{markers_p[p]}-',
                capsize=4, linewidth=2, markersize=8, color=colors_p[p],
                label=f'p={p}')

ax.set_xscale('log')
ax.set_xlabel(r'Noise rate $\gamma$', fontsize=13)
ax.set_ylabel('Peak |C(t*)|', fontsize=13)
ax.set_title(r'Signal Degradation vs Noise (N=8, $\beta$=8, $\mu$=0.1, 30 real.)', fontsize=12)
ax.legend(fontsize=11, title='Sparsity')
ax.set_ylim(0.2, 1.0)
ax.grid(True, alpha=0.3)

# Mark gamma=0 with special tick
ax.set_xticks([5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
ax.set_xticklabels(['0', '0.001', '0.003', '0.01', '0.03', '0.1'])

# Mark 50% line
noiseless_mean = np.mean(peaks[(1.0, 0.0)])
ax.axhline(noiseless_mean * 0.5, color='gray', linestyle=':', alpha=0.5,
           label=f'50% of clean ({noiseless_mean*0.5:.3f})')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '08_gap2_signal_vs_noise.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_gap2_signal_vs_noise.png")

# === Figure 2: Heatmap of peak height in (p, gamma) plane ===
fig, ax = plt.subplots(figsize=(9, 5))

mean_grid = np.zeros((len(sparsity_values), len(gamma_values)))
for i, p in enumerate(sparsity_values):
    for j, gamma in enumerate(gamma_values):
        mean_grid[i, j] = np.mean(peaks[(p, gamma)])

im = ax.imshow(mean_grid, aspect='auto', cmap='viridis', origin='lower',
               vmin=0.3, vmax=0.95)
ax.set_xticks(range(len(gamma_values)))
ax.set_xticklabels([f'{g}' for g in gamma_values], fontsize=10)
ax.set_yticks(range(len(sparsity_values)))
ax.set_yticklabels([f'{p}' for p in sparsity_values], fontsize=11)
ax.set_xlabel(r'Noise rate $\gamma$', fontsize=13)
ax.set_ylabel('Sparsity p', fontsize=13)
ax.set_title('Peak Height Heatmap (Noise x Sparsity)', fontsize=12)

# Annotate cells
for i in range(len(sparsity_values)):
    for j in range(len(gamma_values)):
        val = mean_grid[i, j]
        color = 'white' if val < 0.6 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax, label='Peak |C(t*)|')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '08_gap2_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_gap2_heatmap.png")

# === Figure 3: Normalized signal (ratio to noiseless) showing factorization ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: ratio to noiseless for each p
ax = axes[0]
gamma_nonzero = [g for g in gamma_values if g > 0]
for p in sparsity_values:
    noiseless = np.mean(peaks[(p, 0.0)])
    ratios = [np.mean(peaks[(p, g)]) / noiseless for g in gamma_nonzero]
    ax.plot(gamma_nonzero, ratios, f'{markers_p[p]}-', linewidth=2,
            markersize=8, color=colors_p[p], label=f'p={p}')

ax.set_xscale('log')
ax.set_xlabel(r'Noise rate $\gamma$', fontsize=13)
ax.set_ylabel('Signal / Clean Signal', fontsize=13)
ax.set_title('Signal Retention vs Noise\n(normalized to each p\'s clean value)', fontsize=11)
ax.legend(fontsize=10, title='Sparsity')
ax.set_ylim(0.3, 1.05)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
ax.grid(True, alpha=0.3)

# Right: SEM ratio showing fluctuation growth
ax = axes[1]
for p in sparsity_values:
    sems = [np.std(peaks[(p, g)]) / np.sqrt(len(peaks[(p, g)])) for g in gamma_values]
    gamma_plot = [g if g > 0 else 5e-4 for g in gamma_values]
    ax.plot(gamma_plot, sems, f'{markers_p[p]}-', linewidth=2,
            markersize=8, color=colors_p[p], label=f'p={p}')

ax.set_xscale('log')
ax.set_xlabel(r'Noise rate $\gamma$', fontsize=13)
ax.set_ylabel('SEM of peak height', fontsize=13)
ax.set_title('Disorder Fluctuations vs Noise', fontsize=11)
ax.legend(fontsize=10, title='Sparsity')
ax.set_xticks([5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
ax.set_xticklabels(['0', '0.001', '0.003', '0.01', '0.03', '0.1'])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '08_gap2_factorization.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_gap2_factorization.png")

# === Figure 4: gamma* vs sparsity ===
fig, ax = plt.subplots(figsize=(7, 5))

gamma_stars = []
for p in sparsity_values:
    noiseless_mean = np.mean(peaks[(p, 0.0)])
    half_target = noiseless_mean * 0.5
    gamma_star = None
    for i, gamma in enumerate(gamma_values[1:], 1):
        noisy_mean = np.mean(peaks[(p, gamma)])
        if noisy_mean < half_target:
            prev_gamma = gamma_values[i-1]
            prev_mean = np.mean(peaks[(p, prev_gamma)])
            if prev_mean > half_target:
                frac = (prev_mean - half_target) / (prev_mean - noisy_mean)
                gamma_star = prev_gamma * (gamma / prev_gamma) ** frac
            break
    gamma_stars.append(gamma_star if gamma_star else gamma_values[-1])

ax.plot(sparsity_values, gamma_stars, 'ko-', linewidth=2, markersize=10)
ax.set_xscale('log')
ax.set_xlabel('Sparsity p', fontsize=13)
ax.set_ylabel(r'$\gamma^*$ (50% degradation threshold)', fontsize=13)
ax.set_title(r'Noise Threshold $\gamma^*$ vs Sparsity', fontsize=12)
ax.invert_xaxis()
ax.set_ylim(0.04, 0.07)
ax.grid(True, alpha=0.3)

# Annotate with values
for i, (p, gs) in enumerate(zip(sparsity_values, gamma_stars)):
    ax.annotate(f'{gs:.4f}', (p, gs), textcoords="offset points",
                xytext=(0, 12), fontsize=10, ha='center')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '08_gap2_gamma_star.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_gap2_gamma_star.png")

print("\nAll Gap 2 plots generated.")
