"""
EEG Denoising with SVD -- Synthetic EEG Pipeline
=================================================
Generates synthetic EEG, applies SVD-based denoising (with Gavish-Donoho
adaptive rank selection, multi-channel spatial SVD, and sliding-window SVD),
compares against traditional band-pass and notch filters, and produces
evaluation metrics and plots.
"""

import os
import numpy as np

from eeg_denoising import FS, DURATION, EMBED_DIM, PLOTS_DIR
from eeg_denoising.generation import generate_multichannel_eeg
from eeg_denoising.denoising import (
    build_hankel,
    gavish_donoho_threshold,
    svd_denoise_fixed_k,
    svd_denoise_adaptive,
    multichannel_svd_denoise,
    sliding_window_svd,
    svd_denoise_ml,
    bandpass_filter,
    notch_filter,
)
from eeg_denoising.ml_helpers import train_svd_classifier
from eeg_denoising.metrics import evaluate
from eeg_denoising.plotting import (
    plot_time_comparison,
    plot_singular_values,
    plot_psd_comparison,
    plot_metrics_bar,
    plot_rank_sensitivity,
)

os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    np.random.seed(42)
    n_samples = FS * DURATION
    t = np.linspace(0, DURATION, n_samples, endpoint=False)

    # -- Phase 2: Generate data ------------------------------------------------
    print("Phase 2 -- Generating synthetic EEG ...")
    clean_multi, noisy_multi = generate_multichannel_eeg(t)
    # Use channel 0 for single-channel analyses
    s = clean_multi[0]
    x = noisy_multi[0]

    # -- Phase 3: Denoise ------------------------------------------------------
    print("Phase 3 -- Applying denoising methods ...")

    # Train ML classifier
    print("  Training ML classifier ...")
    clf = train_svd_classifier(fs=FS)

    # 1. Fixed-k SSA (k=2)
    x_svd_fixed, S_fixed = svd_denoise_fixed_k(x, k=2)
    print("  [1/7] Fixed-k SVD (k=2)")

    # 2. Adaptive Gavish-Donoho SSA
    x_svd_adaptive, S_adapt, k_auto = svd_denoise_adaptive(x)
    print(f"  [2/7] Adaptive SVD (auto k={k_auto})")

    # 3. Multi-channel spatial SVD
    noisy_mc_denoised, S_spatial, k_spatial = multichannel_svd_denoise(noisy_multi)
    x_mc = noisy_mc_denoised[0]
    print(f"  [3/7] Multi-channel SVD (auto k={k_spatial})")

    # 4. Sliding-window SVD
    x_sliding = sliding_window_svd(x)
    print("  [4/7] Sliding-window SVD")

    # 5. ML-assisted SVD
    x_ml, S_ml, kept_ml = svd_denoise_ml(x, clf, fs=FS)
    print(f"  [5/7] ML-assisted SVD (kept {len(kept_ml)} components: {list(kept_ml)})")

    # 6. Band-pass filter
    x_bandpass = bandpass_filter(x)
    print("  [6/7] Band-pass filter (1-40 Hz)")

    # 7. Notch filter
    x_notch = notch_filter(x)
    print("  [7/7] Notch filter (60 Hz)")

    # -- Phase 4: Evaluate -----------------------------------------------------
    print("\nPhase 4 -- Computing metrics ...")
    results = [
        evaluate(s, x, x,              "Noisy (raw)"),
        evaluate(s, x, x_svd_fixed,    "SVD fixed k=2"),
        evaluate(s, x, x_svd_adaptive, f"SVD adaptive k={k_auto}"),
        evaluate(s, x, x_mc,           f"Multi-ch SVD k={k_spatial}"),
        evaluate(s, x, x_sliding,      "Sliding-win SVD"),
        evaluate(s, x, x_ml,           "ML-assisted SVD"),
        evaluate(s, x, x_bandpass,     "Band-pass 1-40"),
        evaluate(s, x, x_notch,        "Notch 60 Hz"),
    ]

    # Print table
    header = f"{'Method':<22} {'MSE':>8} {'SNR(dB)':>8} {'Corr':>8} {'dSNR(dB)':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r['Method']:<22} {r['MSE']:8.4f} {r['SNR (dB)']:8.2f} "
              f"{r['Correlation']:8.4f} {r['dSNR (dB)']:9.2f}")

    # -- Plots -----------------------------------------------------------------
    print("\nGenerating plots ...")

    plot_time_comparison(
        t,
        [x, x_svd_adaptive, x_ml, x_bandpass, s],
        ["Noisy", f"SVD adaptive (k={k_auto})", "ML-assisted SVD", "Band-pass", "True clean"],
        "Signal Comparison - Synthetic EEG (first 2 s)",
        "signal_comparison.png",
    )

    plot_time_comparison(
        t,
        [x, x_sliding, x_mc, x_ml, s],
        ["Noisy", "Sliding-window SVD", "Multi-ch SVD", "ML-assisted SVD", "True clean"],
        "Novel Methods - Synthetic EEG (first 2 s)",
        "novel_methods_comparison.png",
    )

    # Singular value spectra
    H_tmp = build_hankel(x, EMBED_DIM)
    m_tmp, n_tmp = H_tmp.shape
    gd_thresh = gavish_donoho_threshold(S_adapt, m_tmp, n_tmp)
    plot_singular_values(S_adapt, "Singular Values - SSA (Single Channel)",
                         "singular_values_ssa.png", threshold=gd_thresh)
    plot_singular_values(S_spatial, "Singular Values - Multi-Channel Spatial SVD",
                         "singular_values_spatial.png")

    # PSD
    plot_psd_comparison(
        [x, x_svd_adaptive, x_ml, x_bandpass, x_notch, s],
        ["Noisy", "SVD adaptive", "ML-assisted SVD", "Band-pass", "Notch 60 Hz", "True clean"],
        FS,
        "Power Spectral Density - Method Comparison",
        "psd_comparison.png",
    )

    # Metrics bar chart
    plot_metrics_bar(results[1:], "metrics_bar.png")  # skip raw

    # Rank-sensitivity
    plot_rank_sensitivity(s, x)

    print("\n[DONE] All done.  Plots saved to:", os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    main()
