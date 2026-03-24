"""
EEG Denoising with SVD — Synthetic EEG Pipeline
=================================================
Generates synthetic EEG, applies SVD-based denoising (with Gavish-Donoho
adaptive rank selection, multi-channel spatial SVD, and sliding-window SVD),
compares against traditional band-pass and notch filters, and produces
evaluation metrics and plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# ─── Configuration ───────────────────────────────────────────────────────────
FS = 256            # Sampling rate (Hz)
DURATION = 10       # Duration (seconds)
N_CHANNELS = 8      # Number of synthetic EEG channels
EMBED_DIM = 50      # Embedding dimension for Hankel/SSA matrix
WINDOW_SEC = 2      # Sliding-window length in seconds
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Synthetic EEG Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_clean_signal(t):
    """Alpha (10 Hz) + Beta (20 Hz) brain-wave mix."""
    alpha = np.sin(2 * np.pi * 10 * t)
    beta = 0.5 * np.sin(2 * np.pi * 20 * t)
    return alpha + beta


def generate_multichannel_eeg(t, n_channels=N_CHANNELS, noise_std=0.5,
                               powerline_amp=0.3):
    """
    Generate *n_channels* of synthetic EEG.

    Each channel shares the same underlying clean signal (with a small random
    phase offset per channel) plus independent Gaussian noise and a common
    60 Hz powerline artifact.
    """
    n_samples = len(t)
    clean = np.zeros((n_channels, n_samples))
    noisy = np.zeros((n_channels, n_samples))

    powerline = powerline_amp * np.sin(2 * np.pi * 60 * t)

    for ch in range(n_channels):
        phase = np.random.rand() * 0.3  # small random phase shift
        s = np.sin(2 * np.pi * 10 * (t + phase)) + \
            0.5 * np.sin(2 * np.pi * 20 * (t + phase))
        noise = noise_std * np.random.randn(n_samples)
        clean[ch] = s
        noisy[ch] = s + noise + powerline

    return clean, noisy


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — Denoising Methods
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_hankel(x, L):
    """Build a Hankel (trajectory) matrix from a 1-D signal x with window L."""
    N = len(x)
    K = N - L + 1
    H = np.zeros((L, K))
    for i in range(L):
        H[i] = x[i:i + K]
    return H


def reconstruct_from_hankel(H, N):
    """Average anti-diagonals of a Hankel matrix to recover a 1-D signal."""
    L, K = H.shape
    x = np.zeros(N)
    counts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            x[i + j] += H[i, j]
            counts[i + j] += 1
    return x / counts


def gavish_donoho_threshold(S, m, n):
    """
    Gavish-Donoho optimal hard threshold for singular values.

    For a matrix of shape (m, n) corrupted by i.i.d. Gaussian noise,
    the optimal threshold is:  ω(β) * √(median(σ²))
    where β = m/n and ω(β) ≈ 0.56 β³ − 0.95 β² + 1.82 β + 1.43.

    Reference: Gavish & Donoho, "The Optimal Hard Threshold for Singular
    Values is 4/√3", IEEE Trans. Inf. Theory, 2014.
    """
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    sigma = np.median(S) / 0.6745  # robust noise-level estimate
    threshold = omega * sigma
    return threshold


# ─── Method 1: Single-Channel SVD (SSA) with fixed k ─────────────────────────

def svd_denoise_fixed_k(x, L=EMBED_DIM, k=2):
    """Singular Spectrum Analysis with a fixed number of components k."""
    N = len(x)
    H = build_hankel(x, L)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    H_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return reconstruct_from_hankel(H_denoised, N), S


# ─── Method 2: SVD with Gavish-Donoho adaptive rank ──────────────────────────

def svd_denoise_adaptive(x, L=EMBED_DIM):
    """SVD denoising with automatic rank selection via Gavish-Donoho."""
    N = len(x)
    H = build_hankel(x, L)
    m, n = H.shape
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    threshold = gavish_donoho_threshold(S, m, n)
    k = int(np.sum(S > threshold))
    k = max(k, 1)  # keep at least one component
    H_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return reconstruct_from_hankel(H_denoised, N), S, k


# ─── Method 3: Multi-Channel Spatial SVD ─────────────────────────────────────

def multichannel_svd_denoise(X, k=None):
    """
    Spatial SVD across channels.

    X : ndarray of shape (n_channels, n_samples)
    Performs SVD on X and retains the top-k components.  If k is None,
    Gavish-Donoho thresholding is used.
    """
    m, n = X.shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if k is None:
        threshold = gavish_donoho_threshold(S, m, n)
        k = int(np.sum(S > threshold))
        k = max(k, 1)
    X_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return X_denoised, S, k


# ─── Method 4: Sliding-Window SVD ────────────────────────────────────────────

def sliding_window_svd(x, fs=FS, window_sec=WINDOW_SEC, L=EMBED_DIM):
    """
    Apply SVD denoising in overlapping sliding windows so that the
    Gavish-Donoho threshold adapts to local noise conditions.
    """
    N = len(x)
    win_len = int(window_sec * fs)
    hop = win_len // 2
    denoised = np.zeros(N)
    weights = np.zeros(N)

    for start in range(0, N - win_len + 1, hop):
        end = start + win_len
        segment = x[start:end]
        seg_dn, _, _ = svd_denoise_adaptive(segment, L=min(L, win_len // 2))
        denoised[start:end] += seg_dn
        weights[start:end] += 1.0

    # Handle any remaining tail
    mask = weights > 0
    denoised[mask] /= weights[mask]
    # Fill un-covered samples (if any) with original signal
    denoised[~mask] = x[~mask]
    return denoised


# ─── Traditional Baselines ───────────────────────────────────────────────────

def bandpass_filter(x, low=1, high=40, fs=FS, order=4):
    """Butterworth band-pass filter."""
    sos = sig.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return sig.sosfiltfilt(sos, x)


def notch_filter(x, freq=60, Q=30, fs=FS):
    """IIR notch filter at the given frequency."""
    b, a = sig.iirnotch(freq, Q, fs=fs)
    return sig.filtfilt(b, a, x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def mse(true, estimated):
    return np.mean((true - estimated) ** 2)


def snr(true, estimated):
    noise_power = np.sum((true - estimated) ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(np.sum(true ** 2) / noise_power)


def correlation(true, estimated):
    return np.corrcoef(true, estimated)[0, 1]


def snr_improvement(true, noisy, denoised):
    return snr(true, denoised) - snr(true, noisy)


def evaluate(true, noisy, denoised, label):
    """Return a dict of all metrics for one method."""
    return {
        "Method": label,
        "MSE": mse(true, denoised),
        "SNR (dB)": snr(true, denoised),
        "Correlation": correlation(true, denoised),
        "ΔSNR (dB)": snr_improvement(true, noisy, denoised),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting Helpers
# ═══════════════════════════════════════════════════════════════════════════════

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
    print(f"  → saved {fname}")


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
    print(f"  → saved {fname}")


def plot_psd_comparison(signals, labels, fs, title, fname):
    """Welch PSD for several signals."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for s, lab in zip(signals, labels):
        f, Pxx = sig.welch(s, fs=fs, nperseg=min(256, len(s)))
        ax.semilogy(f, Pxx, label=lab, alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, fs / 2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  → saved {fname}")


def plot_metrics_bar(results, fname):
    """Bar chart comparing MSE, SNR, Correlation across methods."""
    methods = [r["Method"] for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key in zip(axes, ["MSE", "SNR (dB)", "Correlation", "ΔSNR (dB)"]):
        vals = [r[key] for r in results]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(vals)))
        ax.bar(methods, vals, color=colors)
        ax.set_title(key)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    fig.suptitle("Method Comparison — Synthetic EEG", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {fname}")


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
    print(f"  → saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    n_samples = FS * DURATION
    t = np.linspace(0, DURATION, n_samples, endpoint=False)

    # ── Phase 2: Generate data ────────────────────────────────────────────────
    print("Phase 2 — Generating synthetic EEG …")
    clean_multi, noisy_multi = generate_multichannel_eeg(t)
    # Use channel 0 for single-channel analyses
    s = clean_multi[0]
    x = noisy_multi[0]

    # ── Phase 3: Denoise ──────────────────────────────────────────────────────
    print("Phase 3 — Applying denoising methods …")

    # 1. Fixed-k SSA (k=2)
    x_svd_fixed, S_fixed = svd_denoise_fixed_k(x, k=2)
    print("  [1/6] Fixed-k SVD (k=2)")

    # 2. Adaptive Gavish-Donoho SSA
    x_svd_adaptive, S_adapt, k_auto = svd_denoise_adaptive(x)
    print(f"  [2/6] Adaptive SVD (auto k={k_auto})")

    # 3. Multi-channel spatial SVD
    noisy_mc_denoised, S_spatial, k_spatial = multichannel_svd_denoise(noisy_multi)
    x_mc = noisy_mc_denoised[0]
    print(f"  [3/6] Multi-channel SVD (auto k={k_spatial})")

    # 4. Sliding-window SVD
    x_sliding = sliding_window_svd(x)
    print("  [4/6] Sliding-window SVD")

    # 5. Band-pass filter
    x_bandpass = bandpass_filter(x)
    print("  [5/6] Band-pass filter (1–40 Hz)")

    # 6. Notch filter
    x_notch = notch_filter(x)
    print("  [6/6] Notch filter (60 Hz)")

    # ── Phase 4: Evaluate ─────────────────────────────────────────────────────
    print("\nPhase 4 — Computing metrics …")
    results = [
        evaluate(s, x, x,              "Noisy (raw)"),
        evaluate(s, x, x_svd_fixed,    "SVD fixed k=2"),
        evaluate(s, x, x_svd_adaptive, f"SVD adaptive k={k_auto}"),
        evaluate(s, x, x_mc,           f"Multi-ch SVD k={k_spatial}"),
        evaluate(s, x, x_sliding,      "Sliding-win SVD"),
        evaluate(s, x, x_bandpass,     "Band-pass 1–40"),
        evaluate(s, x, x_notch,        "Notch 60 Hz"),
    ]

    # Print table
    header = f"{'Method':<22} {'MSE':>8} {'SNR(dB)':>8} {'Corr':>8} {'ΔSNR(dB)':>9}"
    print("\n" + header)
    print("─" * len(header))
    for r in results:
        print(f"{r['Method']:<22} {r['MSE']:8.4f} {r['SNR (dB)']:8.2f} "
              f"{r['Correlation']:8.4f} {r['ΔSNR (dB)']:9.2f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_time_comparison(
        t,
        [x, x_svd_adaptive, x_bandpass, s],
        ["Noisy", f"SVD adaptive (k={k_auto})", "Band-pass", "True clean"],
        "Signal Comparison — Synthetic EEG (first 2 s)",
        "signal_comparison.png",
    )

    plot_time_comparison(
        t,
        [x, x_sliding, x_mc, s],
        ["Noisy", "Sliding-window SVD", "Multi-ch SVD", "True clean"],
        "Novel Methods — Synthetic EEG (first 2 s)",
        "novel_methods_comparison.png",
    )

    # Singular value spectra
    # Recompute threshold for annotation
    H_tmp = build_hankel(x, EMBED_DIM)
    m_tmp, n_tmp = H_tmp.shape
    gd_thresh = gavish_donoho_threshold(S_adapt, m_tmp, n_tmp)
    plot_singular_values(S_adapt, "Singular Values — SSA (Single Channel)",
                         "singular_values_ssa.png", threshold=gd_thresh)
    plot_singular_values(S_spatial, "Singular Values — Multi-Channel Spatial SVD",
                         "singular_values_spatial.png")

    # PSD
    plot_psd_comparison(
        [x, x_svd_adaptive, x_bandpass, x_notch, s],
        ["Noisy", "SVD adaptive", "Band-pass", "Notch 60 Hz", "True clean"],
        FS,
        "Power Spectral Density — Method Comparison",
        "psd_comparison.png",
    )

    # Metrics bar chart
    plot_metrics_bar(results[1:], "metrics_bar.png")  # skip raw

    # Rank-sensitivity
    plot_rank_sensitivity(s, x)

    print("\n✓ All done.  Plots saved to:", os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    main()
