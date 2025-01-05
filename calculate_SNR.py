# Import Libraries
import numpy as np
import pandas as pd
import os

def detect_event_regions(clean_signal, threshold=0):
    """
    Detect event regions where the clean signal significantly deviates from zero.
    This assumes that events are regions with significant activity in the clean signal.

    Args:
        clean_signal (array): The clean signal data.
        threshold (float): Threshold to identify non-zero activity in the clean signal.

    Returns:
        events (list): List of tuples with start and end indices for each event region.
    """
    event_indices = np.where(np.abs(clean_signal) > threshold)[0]  # Detect where signal exceeds threshold
    if len(event_indices) == 0:
        return []  # No events detected

    events = []
    start = event_indices[0]
    for i in range(1, len(event_indices)):
        if event_indices[i] != event_indices[i - 1] + 1:  # Break in continuity
            events.append((start, event_indices[i - 1]))
            start = event_indices[i]
    events.append((start, event_indices[-1]))  # Add the last event
    return events

def calculate_event_snr(clean, noisy, events):
    """
    Calculate SNR for identified event regions.

    Args:
        clean (array): The clean signal data.
        noisy (array): The noisy signal data.
        events (list): List of tuples with start and end indices for each event region.

    Returns:
        float: SNR value for the event regions.
    """
    signal_power = 0
    noise_power = 0
    for start, end in events:
        event_clean = clean[start:end+1]
        event_noisy = noisy[start:end+1]
        signal_power += np.sum(event_clean**2)
        noise_power += np.sum((event_noisy - event_clean)**2)

    # Normalize by number of samples and calculate SNR
    signal_power /= len(clean)
    noise_power /= len(clean)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def load_csv(folder):
    """
    Load clean and noisy data from CSV files in the folder.
    """
    clean_data = []
    noisy_data = []

    # Define paths to clean and noisy subdirectories
    clean_folder = os.path.join(folder, "clean")
    noisy_folder = os.path.join(folder, "noisy")

    # Load clean data
    if os.path.exists(clean_folder):
        for file in sorted(os.listdir(clean_folder)):
            if file.endswith(".csv"):
                full_path = os.path.join(clean_folder, file)
                clean_data.append(pd.read_csv(full_path, skiprows=1, header=None).astype(float).values.flatten())
                print(f"Loaded clean file: {full_path}")

    # Load noisy data
    if os.path.exists(noisy_folder):
        for file in sorted(os.listdir(noisy_folder)):
            if file.endswith(".csv"):
                full_path = os.path.join(noisy_folder, file)
                noisy_data.append(pd.read_csv(full_path, skiprows=1, header=None).astype(float).values.flatten())
                print(f"Loaded noisy file: {full_path}")

    return clean_data, noisy_data

# Main script
if __name__ == "__main__":
    # Parameters
    folder = './data/noiselevel1e-4/'  # Folder containing data

    clean_data, noisy_data = load_csv(folder)
    num_files = len(clean_data)
    SNRs = []

    # Process each file
    for count, (clean, sig) in enumerate(zip(clean_data, noisy_data), start=1):
        # Detect events
        events = detect_event_regions(clean, threshold=5e-6)

        if events:
            # Calculate SNR using identified events
            snr_value = calculate_event_snr(clean, sig, events)
            SNRs.append(snr_value)
        else:
            SNRs.append(np.nan)  # Append NaN if no events are found

        print(f"{count}/{num_files} processed.")

    # Final SNR
    SNR_total = np.nanmean(SNRs)  # Use nanmean to skip NaNs if no events exist

    # Save results
    file_name = f"{folder}_SNR_Value.npy"
    np.save(file_name, SNR_total)
    print(f"SNR Total: {SNR_total} saved to {file_name}")
