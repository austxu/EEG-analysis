"""
SVD-based denoising methods and traditional filter baselines.

Methods
-------
- svd_denoise_fixed_k : SSA with a fixed rank k
- svd_denoise_adaptive : SSA with Gavish-Donoho automatic rank
- multichannel_svd_denoise : Spatial SVD across channels
- sliding_window_svd : Locally adaptive SVD in overlapping windows
- bandpass_filter : Butterworth band-pass
- notch_filter : IIR notch at a specified frequency
"""

import numpy as np
from scipy import signal as sig

# Import defaults from package config
from . import FS, EMBED_DIM, WINDOW_SEC


# ---------------------------------------------------------------------------
#  Hankel / SSA helpers
# ---------------------------------------------------------------------------

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

    For an (m, n) matrix corrupted by i.i.d. Gaussian noise the optimal
    threshold is  w(beta) * median(S) / 0.6745  where
    beta = min(m,n)/max(m,n) and
    w(beta) = 0.56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43.

    Reference: Gavish & Donoho, IEEE Trans. Inf. Theory, 2014.
    """
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    sigma = np.median(S) / 0.6745  # robust noise-level estimate
    return omega * sigma


# ---------------------------------------------------------------------------
#  SVD methods
# ---------------------------------------------------------------------------

def svd_denoise_fixed_k(x, L=EMBED_DIM, k=2):
    """Singular Spectrum Analysis with a fixed number of components k."""
    N = len(x)
    H = build_hankel(x, L)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    H_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return reconstruct_from_hankel(H_denoised, N), S


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
    denoised[~mask] = x[~mask]
    return denoised


# ---------------------------------------------------------------------------
#  Traditional filter baselines
# ---------------------------------------------------------------------------

def bandpass_filter(x, low=1, high=40, fs=FS, order=4):
    """Butterworth band-pass filter."""
    sos = sig.butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return sig.sosfiltfilt(sos, x)


def notch_filter(x, freq=60, Q=30, fs=FS):
    """IIR notch filter at the given frequency."""
    b, a = sig.iirnotch(freq, Q, fs=fs)
    return sig.filtfilt(b, a, x)
