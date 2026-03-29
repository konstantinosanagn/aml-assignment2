"""Shared utilities: random seed, plot styling, timing."""

import time
import functools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
FIGURES_DIR = "figures"

def set_global_seed(seed=RANDOM_STATE):
    """Set random seed for reproducibility."""
    np.random.seed(seed)

def setup_plotting():
    """Configure matplotlib/seaborn for consistent, clean plots."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

def save_figure(fig, filename):
    """Save figure to figures/ directory."""
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path)
    print(f"Saved: {path}")
    return path

def timer(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} completed in {elapsed:.2f}s")
        return result, elapsed
    return wrapper
