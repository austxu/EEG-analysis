"""Synthetic EEG signal generation."""

import numpy as np


def generate_clean_signal(t):
    """Alpha (10 Hz) + Beta (20 Hz) brain-wave mix."""
    alpha = np.sin(2 * np.pi * 10 * t)
    beta = 0.5 * np.sin(2 * np.pi * 20 * t)
    return alpha + beta


def generate_multichannel_eeg(t, n_channels=8, noise_std=0.5,
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
