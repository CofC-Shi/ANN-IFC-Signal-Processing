# Import
import numpy as np
import pandas as pd
import scipy.signal as ss
import json
import os
import pickle
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from utils import TimeSeriesDataset, generate_sliding_window_dataset_TSF, read_and_stack_csv_files, apply_delay


# Define processing methods
def apply_notch_filter(signal, fs, frequencies, Q=5):
    filtered_signal = signal.copy()
    for freq in frequencies:
        b, a = ss.iirnotch(freq, Q, fs)
        filtered_signal = ss.filtfilt(b, a, filtered_signal)
    return filtered_signal

def apply_savgol_filter(signal, window_length, polyorder):
    return ss.savgol_filter(signal, window_length, polyorder)

def apply_neural_network_denoising(X_test, model_path, scaler_x_path, scaler_y_path, device="cpu"):
    """
    Apply a trained neural network model for signal denoising.
    """
    # Load the saved model
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}.")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")

    # Load the saved scalers
    if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
        with open(scaler_x_path, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        print(f"Scalers loaded from {scaler_x_path} and {scaler_y_path}.")
    else:
        raise FileNotFoundError("Scalers not found. Please ensure the scalers are saved during training.")

    X_test = scaler_X.transform(X_test)

    # Load data in batches
    test_dataset = TimeSeriesDataset(X_test, np.zeros(len(X_test)))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Denoise the signal
    denoised_signal = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).cpu().numpy()
            denoised_signal.append(outputs)

    denoised_signal = np.concatenate(denoised_signal, axis=0)
    denoised_signal = scaler_y.inverse_transform(denoised_signal)
    delayed_denoised_signal = apply_delay(denoised_signal, window_size)

    return delayed_denoised_signal.flatten()

# Noise levels and corresponding datasets
noise_levels = ['2_2e-5', '5e-5', '1e-4', '2e-4', '3e-4', '5e-4', '7e-4']
data_path = 'data/'

# Processing parameters
fs = 7196  # Sampling frequency
notch_freqs = [60, 120]  # Frequencies for notch filtering
window_size = 32
savgol_params = {'window_length': 10, 'polyorder': 3}
model_path = './models/Custom_TF_stream_30_1e-4.pkl'
scaler_x_path = './scalers/Custom_TF_stream_30_1e-4_X.pkl'
scaler_y_path = './scalers/Custom_TF_stream_30_1e-4_y.pkl'

device = "cuda" if torch.cuda.is_available() else "cpu"

processed_signals = {}

for noise_level in noise_levels:
    # Load clean and noisy data
    clean_file = f'{data_path}/noiselevel{noise_level}/clean'
    noisy_file = f'{data_path}/noiselevel{noise_level}/noisy'

    clean_data = read_and_stack_csv_files(clean_file)
    noisy_data = read_and_stack_csv_files(noisy_file)

    # Generate sliding window dataset
    windows_total, next_values_total = generate_sliding_window_dataset_TSF(clean_data, noisy_data, window_size)

    print(f"Shape of windows_total: {windows_total.shape}, Shape of next_values_total: {next_values_total.shape}")

    # Use the entire dataset for testing
    X_test = windows_total
    y_test = next_values_total

    # Apply filters
    tf_signal = apply_notch_filter(noisy_data, fs, notch_freqs)
    sg_signal = apply_savgol_filter(noisy_data, **savgol_params)

    # Apply neural network
    nn_signal = apply_neural_network_denoising(X_test, model_path, scaler_x_path, scaler_y_path, device)

    # Save processed signals
    processed_signals[noise_level] = {
        'clean': clean_data.tolist(),
        'noisy': noisy_data.tolist(),
        'notch': tf_signal.tolist(),
        'savgol': sg_signal.tolist(),
        'neural_network': nn_signal.tolist()
    }

# Save results
output_dir = "./processed_signals_by_noise_level"
os.makedirs(output_dir, exist_ok=True)

for noise_level, data in processed_signals.items():
    output_file = os.path.join(output_dir, f'processed_signals_{noise_level}.json')
    with open(output_file, 'w') as out_f:
        json.dump({noise_level: data}, out_f, indent=4)

print(f"Processed signals split into individual files in {output_dir}")