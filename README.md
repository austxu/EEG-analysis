# EEG Denoising with SVD

An EEG signal denoising pipeline that compares **SVD-based methods** against
traditional filtering baselines, with novel enhancements including
**Gavish-Donoho adaptive rank selection**, **multi-channel spatial SVD**, and
**sliding-window SVD**.

## Quick Start

```bash
pip install -r requirements.txt

# Synthetic EEG pipeline (ground-truth evaluation)
python main.py

# Real PhysioNet EEG pipeline (proxy-metric evaluation)
python analyze_real_data.py
```

Plots are saved to the `plots/` directory.

## Methods

| Method | Description |
|--------|-------------|
| **SVD fixed k** | Singular Spectrum Analysis with a manually chosen rank |
| **SVD adaptive (Gavish-Donoho)** | Automatic rank via optimal hard thresholding |
| **Multi-channel spatial SVD** | SVD across the channel×time matrix for spatial denoising |
| **Sliding-window SVD** | Locally adaptive SVD for non-stationary signals |
| **Band-pass filter** | Butterworth 1–40 Hz |
| **Notch filter** | IIR notch at 60 Hz |

## Evaluation

### Synthetic Data (ground truth available)
- **MSE** — mean squared error (lower is better)
- **SNR** — signal-to-noise ratio in dB (higher is better)
- **Correlation** — Pearson ρ (closer to 1 is better)
- **ΔSNR** — SNR improvement over noisy signal (positive is good)

### Real Data (no ground truth)
- **60 Hz Power Reduction** — dB reduction in the powerline band
- **α/β Preservation** — ratio of alpha+beta power after/before (≈ 1.0 is ideal)

## Analysis & Conclusions

- **SVD (adaptive)** removes noise without assuming fixed frequency content,
  making it more flexible than band-pass or notch filters.
- **Band-pass filtering** is effective but removes all content outside its
  passband, which may distort wideband signals.
- **Notch filtering** targets 60 Hz specifically but leaves broadband noise
  untouched.
- **Gavish-Donoho thresholding** eliminates the need to manually tune *k*,
  adapting automatically to the noise level.
- **Multi-channel SVD** leverages spatial redundancy across electrodes to
  separate brain signals from sensor-specific noise.
- **Sliding-window SVD** handles non-stationarity, adapting the denoising
  threshold to local signal conditions.

## Project Structure

```
eeg_project/
├── eeg_denoising/           # Core Python package
│   ├── __init__.py          # Config + re-exports
│   ├── generation.py        # Synthetic signal generation
│   ├── denoising.py         # SVD methods + traditional filters
│   ├── metrics.py           # MSE, SNR, correlation
│   └── plotting.py          # All visualization helpers
├── main.py                  # Synthetic EEG pipeline (entry point)
├── analyze_real_data.py     # Real EEG pipeline (entry point)
├── requirements.txt
├── README.md
└── plots/                   # Generated figures
```
