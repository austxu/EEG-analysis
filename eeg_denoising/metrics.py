"""Evaluation metrics for EEG denoising."""

import numpy as np


def mse(true, estimated):
    """Mean squared error."""
    return np.mean((true - estimated) ** 2)


def snr(true, estimated):
    """Signal-to-noise ratio in dB."""
    noise_power = np.sum((true - estimated) ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(np.sum(true ** 2) / noise_power)


def correlation(true, estimated):
    """Pearson correlation coefficient."""
    return np.corrcoef(true, estimated)[0, 1]


def snr_improvement(true, noisy, denoised):
    """SNR improvement (dB) relative to the noisy signal."""
    return snr(true, denoised) - snr(true, noisy)


def evaluate(true, noisy, denoised, label):
    """Return a dict of all metrics for one method."""
    return {
        "Method": label,
        "MSE": mse(true, denoised),
        "SNR (dB)": snr(true, denoised),
        "Correlation": correlation(true, denoised),
        "dSNR (dB)": snr_improvement(true, noisy, denoised),
    }
