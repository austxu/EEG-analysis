"""
EEG Denoising with SVD
======================
A package for EEG signal denoising using SVD-based methods
(Gavish-Donoho adaptive rank, multi-channel spatial SVD,
sliding-window SVD) and traditional filtering baselines.
"""

# Configuration defaults
FS = 256            # Sampling rate (Hz)
DURATION = 10       # Duration (seconds)
N_CHANNELS = 8      # Number of synthetic EEG channels
EMBED_DIM = 50      # Embedding dimension for Hankel/SSA matrix
WINDOW_SEC = 2      # Sliding-window length in seconds
PLOTS_DIR = "plots"

from .generation import generate_clean_signal, generate_multichannel_eeg
from .denoising import (
    build_hankel,
    reconstruct_from_hankel,
    gavish_donoho_threshold,
    svd_denoise_fixed_k,
    svd_denoise_adaptive,
    multichannel_svd_denoise,
    sliding_window_svd,
    svd_denoise_ml,
    bandpass_filter,
    notch_filter,
)
from .ml_helpers import train_svd_classifier, extract_component_features
from .metrics import mse, snr, correlation, snr_improvement, evaluate
from .plotting import (
    plot_time_comparison,
    plot_singular_values,
    plot_psd_comparison,
    plot_metrics_bar,
    plot_rank_sensitivity,
)
