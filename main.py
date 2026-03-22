import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import mne

def generate_dummy_eeg_data(duration_sec=10, sampling_rate=256, n_channels=4):
    """
    Generates some dummy EEG data for initial analysis testing.
    """
    n_samples = duration_sec * sampling_rate
    time = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    # Generate some synthetic signals (mix of sine waves and noise)
    data = []
    for i in range(n_channels):
        # Base noise
        noise = np.random.normal(0, 5, n_samples)
        
        # Alpha wave (8-12 Hz)
        alpha = 10 * np.sin(2 * np.pi * 10 * time + np.random.rand() * 2 * np.pi)
        
        # Beta wave (12-30 Hz)
        beta = 5 * np.sin(2 * np.pi * 20 * time + np.random.rand() * 2 * np.pi)
        
        # Combine
        channel_data = noise + alpha + beta
        data.append(channel_data)
        
    data = np.array(data)
    
    # Create an MNE Info object and RawArray
    ch_names = [f'EEG {i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    return raw

if __name__ == "__main__":
    print("Generating dummy EEG data...")
    raw = generate_dummy_eeg_data()
    
    print(raw.info)
    
    # Example analysis: Plotting the power spectral density (PSD)
    # raw.compute_psd().plot()
    # plt.show()
    
    print("Done. Ready for analysis.")
