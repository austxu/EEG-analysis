"""
ML-assisted SVD component classification.

Trains a lightweight RandomForestClassifier to distinguish brain-signal
SVD components from artifact components (powerline hum, broadband noise)
based on spectral and statistical features.
"""

import numpy as np
from scipy import signal as sig
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
#  Feature extraction
# ---------------------------------------------------------------------------

def extract_component_features(component, fs):
    """
    Extract discriminative features from a single reconstructed SVD component.

    Features
    --------
    1. variance           – overall energy
    2. peak_frequency     – dominant frequency via Welch PSD
    3. powerline_ratio    – fraction of power in 55-65 Hz band
    4. alpha_beta_ratio   – fraction of power in 8-30 Hz (brain) band
    5. spectral_entropy   – Shannon entropy of normalised PSD
    6. zero_crossing_rate – proxy for dominant oscillation frequency
    7. kurtosis           – peakedness (spiky artifacts vs smooth waves)
    """
    n = len(component)
    nperseg = min(256, n)

    # Welch PSD
    f, Pxx = sig.welch(component, fs=fs, nperseg=nperseg)
    Pxx_sum = Pxx.sum()
    if Pxx_sum == 0:
        Pxx_norm = np.ones_like(Pxx) / len(Pxx)
    else:
        Pxx_norm = Pxx / Pxx_sum

    # 1. Variance
    variance = np.var(component)

    # 2. Peak frequency
    peak_frequency = f[np.argmax(Pxx)]

    # 3. Powerline ratio (55-65 Hz)
    nyquist = fs / 2
    if nyquist > 55:
        pl_mask = (f >= 55) & (f <= min(65, nyquist))
        powerline_ratio = Pxx[pl_mask].sum() / Pxx_sum if Pxx_sum > 0 else 0.0
    else:
        powerline_ratio = 0.0

    # 4. Alpha+Beta ratio (8-30 Hz)
    ab_mask = (f >= 8) & (f <= 30)
    alpha_beta_ratio = Pxx[ab_mask].sum() / Pxx_sum if Pxx_sum > 0 else 0.0

    # 5. Spectral entropy
    Pxx_pos = Pxx_norm[Pxx_norm > 0]
    spectral_entropy = -np.sum(Pxx_pos * np.log2(Pxx_pos))

    # 6. Zero-crossing rate
    signs = np.sign(component - np.mean(component))
    zero_crossing_rate = np.sum(np.abs(np.diff(signs)) > 0) / n

    # 7. Kurtosis
    std = np.std(component)
    if std > 0:
        kurtosis = np.mean(((component - np.mean(component)) / std) ** 4) - 3
    else:
        kurtosis = 0.0

    return np.array([
        variance,
        peak_frequency,
        powerline_ratio,
        alpha_beta_ratio,
        spectral_entropy,
        zero_crossing_rate,
        kurtosis,
    ])


# ---------------------------------------------------------------------------
#  Training data generation
# ---------------------------------------------------------------------------

def _generate_training_set(fs, n_samples=2048, n_examples=500):
    """
    Build a synthetic training set of labelled SVD-like components.

    Returns (X_features, y_labels) where
      label 1 = brain signal (keep)
      label 0 = artifact     (drop)
    """
    rng = np.random.RandomState(0)
    t = np.arange(n_samples) / fs
    X, y = [], []

    for _ in range(n_examples):
        kind = rng.choice(["alpha", "beta", "mixed", "powerline", "noise", "spike"])

        if kind == "alpha":
            freq = rng.uniform(8, 13)
            comp = np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
            comp *= rng.uniform(0.3, 1.5)
            label = 1
        elif kind == "beta":
            freq = rng.uniform(13, 30)
            comp = np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
            comp *= rng.uniform(0.2, 1.0)
            label = 1
        elif kind == "mixed":
            f1 = rng.uniform(8, 13)
            f2 = rng.uniform(13, 30)
            comp = (np.sin(2 * np.pi * f1 * t) +
                    0.5 * np.sin(2 * np.pi * f2 * t))
            comp *= rng.uniform(0.3, 1.2)
            label = 1
        elif kind == "powerline":
            freq = rng.uniform(58, 62)  # slight jitter around 60 Hz
            comp = np.sin(2 * np.pi * freq * t)
            comp *= rng.uniform(0.5, 3.0)
            label = 0
        elif kind == "noise":
            comp = rng.randn(n_samples) * rng.uniform(0.3, 1.5)
            label = 0
        else:  # spike
            comp = np.zeros(n_samples)
            n_spikes = rng.randint(1, 10)
            for _ in range(n_spikes):
                idx = rng.randint(0, n_samples)
                width = rng.randint(2, 20)
                lo = max(0, idx - width)
                hi = min(n_samples, idx + width)
                comp[lo:hi] = rng.uniform(2, 8) * rng.choice([-1, 1])
            label = 0

        # Add a touch of noise to brain signals to be realistic
        if label == 1:
            comp += rng.randn(n_samples) * 0.05

        X.append(extract_component_features(comp, fs))
        y.append(label)

    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
#  Classifier training
# ---------------------------------------------------------------------------

def train_svd_classifier(fs, n_samples=2048, n_examples=500):
    """
    Train and return a RandomForestClassifier for SVD component selection.

    The classifier is trained on synthetically generated examples of brain
    signals and artifacts so it can generalise to real SVD components.
    """
    X, y = _generate_training_set(fs, n_samples=n_samples,
                                  n_examples=n_examples)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                 random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf
