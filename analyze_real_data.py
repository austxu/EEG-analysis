import mne
from mne.datasets import eegbci

def fetch_and_load_real_eeg_data(subject=1, runs=[1]):
    """
    Fetches real open-source EEG data from the PhysioNet BCI dataset using MNE.
    Subject 1, Run 1 is baseline (eyes open).
    """
    print(f"Fetching PhysioNet EEG BCI data for Subject {subject}, Runs {runs}...")
    
    # Download the EDF files to the local mne_data directory
    raw_fnames = eegbci.load_data(subject, runs)
    print(f"Data downloaded/found at: {raw_fnames}")
    
    # Load the EDF file into an MNE Raw object
    # preload=True loads the data into memory
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
    
    # Apply standard 10-05 montage since this dataset uses standard channel names
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    
    return raw

if __name__ == "__main__":
    # Fetch the real open source data and load it
    raw_real = fetch_and_load_real_eeg_data(subject=1, runs=[1])
    
    print("\nReal EEG Data Info:")
    print(raw_real.info)
    
    # Example: Apply a bandpass filter (1 Hz to 40 Hz) common in EEG analysis
    print("\nApplying bandpass filter (1-40 Hz)...")
    raw_filtered = raw_real.copy().filter(l_freq=1.0, h_freq=40.0)
    
    print("Code for fetching and processing real open-source EEG data is ready to use!")
