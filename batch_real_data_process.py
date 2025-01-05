import numpy as np
import os
import scipy.io as sio
from scipy.signal import find_peaks, detrend
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
from utils import (
    apply_neural_network_denoising,
    load_data,
    apply_delay,
)

matplotlib.use("Qt5Agg")  # Use the Qt5Agg backend
def generate_sliding_window(noisy_data, window_size):
    """
    Generates sliding window datasets from noisy data for denoising.
    """
    if len(noisy_data) < window_size:
        raise ValueError("Window size must be smaller than the length of the noisy data.")

    return np.lib.stride_tricks.sliding_window_view(noisy_data, window_size)

def calculate_custom_widths(signal, peak_indices, threshold, sampling_frequency):
    """
    Calculates the width of each peak at a given threshold.

    Args:
        signal (numpy array): The signal data.
        peak_indices (numpy array): Indices of detected peaks.
        threshold (float): The threshold value to find intersections.
        sampling_frequency (float): Sampling frequency in Hz.

    Returns:
        list: A list of widths in milliseconds for each peak.
    """
    widths = []

    for peak_index in peak_indices:
        # Initialize left and right intersection indices
        left_intersection = None
        right_intersection = None

        # Search to the left of the peak
        for i in range(peak_index, 0, -1):
            if signal[i] <= threshold:
                left_intersection = i
                break

        # Search to the right of the peak
        for i in range(peak_index, len(signal)):
            if signal[i] <= threshold:
                right_intersection = i
                break

        # Calculate the width
        if left_intersection is not None and right_intersection is not None:
            width_samples = right_intersection - left_intersection
            width_ms = width_samples / sampling_frequency * 1000  # Convert to milliseconds
            widths.append(width_ms)
        else:
            widths.append(None)  # Append None if width can't be calculated

    return widths

def detect_peaks(processed_data, sampling_frequency, noise_threshold, prominence_threshold):
    """
    Detects positive and negative peaks in the processed data and filters them by prominence.
    """
    # Positive Peaks
    pks_pos, properties_pos = find_peaks(
        processed_data,
        height=noise_threshold,  # Adjusted height threshold for positive peaks
        distance=0.1 * sampling_frequency,  # Minimum distance between peaks in samples
        prominence=prominence_threshold  # Minimum prominence
    )

    locs_pos = pks_pos  # Indices of positive peaks
    p_pos = properties_pos["prominences"]  # Prominence of positive peaks

    # Define a threshold for custom width calculation
    threshold = 0.00001

    # Calculate custom widths for positive peaks
    widths_time_pos = calculate_custom_widths(processed_data, locs_pos, threshold, sampling_frequency)

    # Negative Peaks (invert the signal for detection)
    pks_neg, properties_neg = find_peaks(
        -processed_data,
        height=noise_threshold,  # Minimum height for negative peaks
        distance=0.1 * sampling_frequency,  # Minimum distance between peaks
        prominence=prominence_threshold  # Minimum prominence
    )

    locs_neg = pks_neg  # Indices of negative peaks
    p_neg = properties_neg["prominences"]  # Prominence of negative peaks
    pks_neg = -properties_neg["peak_heights"]  # Revert to original signal values

    # Define a threshold for custom width calculation
    threshold = 0.00001

    # Calculate custom widths for negative peaks
    widths_time_neg = calculate_custom_widths(-processed_data, locs_neg, threshold, sampling_frequency)

    # Convert locations to milliseconds
    locs_pos_ms = locs_pos / sampling_frequency * 1000
    locs_neg_ms = locs_neg / sampling_frequency * 1000

    # Combine positive and negative peaks
    locs_ms = np.concatenate((locs_pos_ms, locs_neg_ms))
    pks_filtered = np.concatenate((processed_data[locs_pos], pks_neg))
    prominences_filtered = np.concatenate((p_pos, p_neg))
    widths_time = np.concatenate((widths_time_pos, widths_time_neg))

    return {
        "locs_ms": locs_ms,
        "pks_filtered": pks_filtered,
        "prominences_filtered": prominences_filtered,
        "widths_time": widths_time,
    }

def plot_peaks(noisy_data, processed_data, result, sampling_frequency, file_name):
    """
    Plots the processed data along with detected peaks.

    Args:
        noisy_data (numpy array): Original noisy data.
        processed_data (numpy array): Denoised data after processing.
        result (dict): Processed peak information.
        sampling_frequency (float): Sampling frequency in Hz.
    """
    # Align time with processed_data length
    time = np.arange(len(processed_data)) / sampling_frequency  # Adjust time to match processed data length
    noisy_time = np.arange(len(noisy_data)) / sampling_frequency  # Time for noisy data

    plt.figure(figsize=(12, 6))

    # Plot noisy data
    plt.plot(noisy_time, noisy_data, label="Noisy Data", alpha=0.5, color="gray")

    # Plot processed (denoised) data
    plt.plot(time, processed_data, label="Denoised Data", alpha=0.8, color="blue")

    # Plot detected peaks
    plt.scatter(
        result["locs_ms"] / 1000,
        result["pks_filtered"],
        color="red",
        label="Detected Peaks",
    )

    # Add labels and legend
    # plt.title(f"Detected Peaks for File: {file_name}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # #

    # save_path = os.path.join("./results", f"{file_name}_plot.png")
    # plt.savefig(save_path)
    # plt.close()

def process_mat_file(file_path, window_size, model_path, scaler_x_path, scaler_y_path, device):
    """
    Processes a single .mat file to extract peaks from denoised signal.
    """
    try:
        data_out, phase, freq_labels, sampling_frequency = load_data(file_path)
        noisy_data = np.array(data_out[1])  # Select desired channel

        # Detrend data
        noisy_data = detrend(noisy_data - np.mean(noisy_data))

        # Generate sliding windows
        X_test = generate_sliding_window(noisy_data, window_size)

        # Denoise data using neural network
        denoised_signal = apply_neural_network_denoising(
            X_test, model_path, scaler_x_path, scaler_y_path, device
        )
        delayed_denoised_signal = apply_delay(denoised_signal, window_size)

        return noisy_data, delayed_denoised_signal, sampling_frequency
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None

def process_folder(folder_path, output_csv_path, window_size, model_path, scaler_x_path, scaler_y_path):
    """
    Processes all .mat files in a folder and saves extracted features to a CSV file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mat"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_path}...")

            noisy_data, processed_signal, sampling_frequency = process_mat_file(
                file_path, window_size, model_path, scaler_x_path, scaler_y_path, device
            )
            if noisy_data is None or processed_signal is None:
                continue

            # Detect peaks
            result = detect_peaks(processed_signal, sampling_frequency, noise_threshold=0.00005, prominence_threshold=1.5e-5)

            # Plot peaks (non-blocking)
            plot_peaks(noisy_data, processed_signal, result, sampling_frequency, file_name)

            # Save extracted features
            for loc, amp, prom, width_time in zip(result["locs_ms"], result["pks_filtered"], result["prominences_filtered"], result["widths_time"]):
                if not width_time:
                    print(f"Width is Nan at location {loc} ms, amplitude: {amp}, prominence: {prom}")
                    # Plot the region around the peak
                    plt.figure(figsize=(12, 6))
                    plt.plot(processed_signal, label="Processed Data")
                    plt.axvline(x=int(loc / 1000 * sampling_frequency), color="red", linestyle="--",
                                label="Peak Location")
                    plt.title(f"Debugging Width at Location {loc} ms")
                    plt.xlabel("Sample Index")
                    plt.ylabel("Amplitude")
                    plt.legend()
                    plt.show()
                features.append({"File Name": file_name, "Location (ms)": loc, "Width time (ms)": width_time, "Amplitude": amp, "Prominence": prom})

    # Save to CSV
    if features:
        df = pd.DataFrame(features)
        df.to_csv(output_csv_path, index=False)
        print(f"Features saved to {output_csv_path}")
    else:
        print("No features extracted. Check input data or processing logic.")

# Configuration
folder_path = "../experimental_data/4um_031224/"
output_csv_path = "extracted_features.csv"
model_path = "./models/Custom_TF_stream_30_7e-4_symmetric.pkl"
scaler_x_path = "./scalers/Custom_TF_stream_30_7e-4_symmetric_X.pkl"
scaler_y_path = "./scalers/Custom_TF_stream_30_7e-4_symmetric_y.pkl"
window_size = 32

# Process folder and save features
process_folder(folder_path, output_csv_path, window_size, model_path, scaler_x_path, scaler_y_path)
