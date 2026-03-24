"""Plotting helpers for EEG denoising results."""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

from . import PLOTS_DIR, EMBED_DIM
from .denoising import build_hankel, reconstruct_from_hankel
from .metrics import mse, snr

os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_time_comparison(t, signals, labels, title, fname):
    """Overlay multiple signals on a shared time axis."""
    fig, ax = plt.subplots(figsize=(14, 4))
    for s, lab in zip(signals, labels):
        ax.plot(t, s, label=lab, alpha=0.8, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(t[0], min(t[-1], 2))  # zoom first 2 s for clarity
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  -> saved {fname}")


def plot_singular_values(S, title, fname, threshold=None):
    """Plot singular-value spectrum."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(S, 'o-', markersize=3)
    if threshold is not None:
        ax.axhline(threshold, color='r', ls='--', label=f"GD threshold = {threshold:.2f}")
        ax.legend()
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  -> saved {fname}")


def plot_psd_comparison(signals, labels, fs, title, fname):
    """Welch PSD for several signals."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for s, lab in zip(signals, labels):
        f, Pxx = sig.welch(s, fs=fs, nperseg=min(256, len(s)))
        ax.semilogy(f, Pxx, label=lab, alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, fs / 2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  -> saved {fname}")


def plot_metrics_bar(results, fname):
    """Bar chart comparing MSE, SNR, Correlation across methods."""
    methods = [r["Method"] for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key in zip(axes, ["MSE", "SNR (dB)", "Correlation", "dSNR (dB)"]):
        vals = [r[key] for r in results]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(vals)))
        ax.bar(methods, vals, color=colors)
        ax.set_title(key)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    fig.suptitle("Method Comparison - Synthetic EEG", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {fname}")


def plot_rank_sensitivity(true, noisy_signal, L=EMBED_DIM, fname="rank_sensitivity.png"):
    """Sweep k and plot MSE / SNR vs rank."""
    N = len(noisy_signal)
    H = build_hankel(noisy_signal, L)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    max_k = min(30, len(S))
    ks = range(1, max_k + 1)
    mses, snrs = [], []
    for k in ks:
        H_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        recon = reconstruct_from_hankel(H_k, N)
        mses.append(mse(true, recon))
        snrs.append(snr(true, recon))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, mses, 'o-', markersize=3)
    ax1.set_xlabel("Rank k")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE vs. SVD Rank")
    ax2.plot(ks, snrs, 's-', markersize=3, color="tab:orange")
    ax2.set_xlabel("Rank k")
    ax2.set_ylabel("SNR (dB)")
    ax2.set_title("SNR vs. SVD Rank")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  -> saved {fname}")
