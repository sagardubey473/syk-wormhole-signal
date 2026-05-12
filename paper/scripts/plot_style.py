"""Shared matplotlib configuration for paper figures."""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(PAPER_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(PAPER_DIR, 'figures')

# Figure dimensions (inches) for REVTeX two-column
SINGLE_COL_WIDTH = 3.375
DOUBLE_COL_WIDTH = 7.0
GOLDEN_RATIO = 1.618


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    try:
        plt.rcParams.update({'text.usetex': True})
        # Test if LaTeX works
        fig_test = plt.figure()
        plt.close(fig_test)
    except Exception:
        plt.rcParams.update({'text.usetex': False,
                             'mathtext.fontset': 'cm'})

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
    })


def ensure_fig_dir():
    """Create figures directory if it doesn't exist."""
    os.makedirs(FIG_DIR, exist_ok=True)


# Colorblind-safe palette (IBM Design)
COLORS = {
    'blue': '#0072B2',
    'orange': '#D55E00',
    'green': '#009E73',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'cyan': '#56B4E9',
    'red': '#E69F00',
    'gray': '#999999',
}

# Sparsity-specific styling
SPARSITY_COLORS = {
    1.0: COLORS['blue'],
    0.5: COLORS['cyan'],
    0.3: COLORS['green'],
    0.2: COLORS['yellow'],
    0.1: COLORS['orange'],
    0.07: COLORS['red'],
    0.05: COLORS['purple'],
    0.03: COLORS['gray'],
    0.02: '#000000',
}

SPARSITY_MARKERS = {
    1.0: 'o', 0.5: 's', 0.3: 'D', 0.2: '^',
    0.1: 'v', 0.07: '<', 0.05: '>', 0.03: 'p', 0.02: '*',
}
