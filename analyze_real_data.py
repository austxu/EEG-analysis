"""
EEG Denoising with SVD — Real EEG Pipeline
============================================
Loads real EEG from the PhysioNet BCI dataset, applies SVD-based denoising
and traditional filters, then evaluates using proxy metrics (PSD analysis,
band-power preservation) since no ground truth is available.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import mne
from mne.datasets import eegbci

# Reuse denoising functions from main.py
from main import (
    svd_denoise_adaptive,
    multichannel_svd_denoise,
    sliding_window_svd,
    bandpass_filter,
    notch_filter,
)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_physionet_eeg(subject=1, runs=[1]):
    """Fetch PhysioNet EEG BCI data and return raw MNE object + numpy arrays."""
    print(f"Fetching PhysioNet BCI data — Subject {subject}, Runs {runs} …")
    raw_fnames = eegbci.load_data(subject, runs)
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    fs = int(raw.info['sfreq'])
    data = raw.get_data()  # (n_channels, n_samples)
    ch_names = raw.ch_names
    return data, fs, ch_names, raw


# ═══════════════════════════════════════════════════════════════════════════════
#  Proxy Metrics (no ground truth)
# ═══════════════════════════════════════════════════════════════════════════════

def band_power(x, fs, low, high):
    """Total power in [low, high] Hz via Welch."""
    f, Pxx = sig.welch(x, fs=fs, nperseg=min(512, len(x)))
    mask = (f >= low) & (f <= high)
    return np.trapz(Pxx[mask], f[mask])


def powerline_reduction_db(original, denoised, fs, freq=60, bw=2):
    """Reduction in power around `freq` Hz (in dB)."""
    p_before = band_power(original, fs, freq - bw, freq + bw)
    p_after = band_power(denoised, fs, freq - bw, freq + bw)
    if p_after == 0:
        return np.inf
    return 10 * np.log10(p_before / p_after)


def alpha_beta_preservation(original, denoised, fs):
    """Ratio of alpha+beta power after/before denoising (ideally ≈ 1.0)."""
    ab_before = band_power(original, fs, 8, 30)
    ab_after = band_power(denoised, fs, 8, 30)
    if ab_before == 0:
        return np.nan
    return ab_after / ab_before


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_real_time(t, signals, labels, title, fname):
    fig, ax = plt.subplots(figsize=(14, 4))
    for s, lab in zip(signals, labels):
        ax.plot(t, s, label=lab, alpha=0.8, linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(t[0], min(t[-1], 3))  # zoom first 3 s
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  → saved {fname}")


def plot_real_psd(signals, labels, fs, title, fname):
    fig, ax = plt.subplots(figsize=(10, 4))
    for s, lab in zip(signals, labels):
        f, Pxx = sig.welch(s, fs=fs, nperseg=min(512, len(s)))
        ax.semilogy(f, Pxx, label=lab, alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(80, fs / 2))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  → saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    data, fs, ch_names, raw = load_physionet_eeg(subject=1, runs=[1])
    n_channels, n_samples = data.shape
    t = np.arange(n_samples) / fs
    print(f"Loaded {n_channels} channels, {n_samples} samples, fs={fs} Hz")

    # Pick a representative channel
    ch_idx = 0
    ch_name = ch_names[ch_idx]
    x = data[ch_idx]
    print(f"\nAnalysing channel: {ch_name}")

    # ── Apply denoising methods ───────────────────────────────────────────────
    print("\nApplying denoising methods …")

    # 1. SVD adaptive (single-channel SSA)
    x_svd, S_svd, k_svd = svd_denoise_adaptive(x, L=50)
    print(f"  [1/4] Adaptive SVD (k={k_svd})")

    # 2. Multi-channel spatial SVD
    data_mc, S_mc, k_mc = multichannel_svd_denoise(data)
    x_mc = data_mc[ch_idx]
    print(f"  [2/4] Multi-channel SVD (k={k_mc})")

    # 3. Sliding-window SVD
    x_sw = sliding_window_svd(x, fs=fs)
    print("  [3/4] Sliding-window SVD")

    # 4. Band-pass + Notch combo
    x_filt = notch_filter(bandpass_filter(x, fs=fs), fs=fs)
    print("  [4/4] Band-pass (1–40 Hz) + Notch (60 Hz)")

    # ── Proxy Metrics ─────────────────────────────────────────────────────────
    print("\nProxy Metrics (no ground truth):")
    methods = {
        "SVD adaptive": x_svd,
        "Multi-ch SVD": x_mc,
        "Sliding-win SVD": x_sw,
        "BP + Notch": x_filt,
    }

    header = f"{'Method':<20} {'60Hz Reduction(dB)':>20} {'α/β Preservation':>18}"
    print(header)
    print("─" * len(header))
    for name, denoised in methods.items():
        plr = powerline_reduction_db(x, denoised, fs)
        abp = alpha_beta_preservation(x, denoised, fs)
        print(f"{name:<20} {plr:>20.2f} {abp:>18.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_real_time(
        t, [x, x_svd, x_filt],
        ["Original", f"SVD adaptive (k={k_svd})", "BP+Notch"],
        f"Real EEG — {ch_name} (first 3 s)",
        "real_signal_comparison.png",
    )

    plot_real_time(
        t, [x, x_sw, x_mc],
        ["Original", "Sliding-win SVD", "Multi-ch SVD"],
        f"Real EEG — Novel Methods — {ch_name} (first 3 s)",
        "real_novel_comparison.png",
    )

    plot_real_psd(
        [x, x_svd, x_sw, x_filt],
        ["Original", "SVD adaptive", "Sliding-win SVD", "BP+Notch"],
        fs,
        f"PSD — Real EEG ({ch_name})",
        "real_psd_comparison.png",
    )

    print("\n✓ All done.  Plots saved to:", os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    main()
