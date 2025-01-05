import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import (
    TimeSeriesDataset,
    apply_neural_network_denoising,
    load_data,
    read_and_stack_csv_files,
    validation_plot_real,
    apply_delay,
)
import matplotlib
import os
from scipy import signal


matplotlib.use("Qt5Agg")  # Use the Qt5Agg backend
def generate_sliding_window(noisy_data, window_size):
    """
    Generates sliding window datasets from noisy data for denoising.
    """
    # Ensure noisy_data is a numpy array
    noisy_data = np.array(noisy_data)

    # Calculate the number of windows
    num_windows = len(noisy_data) - window_size + 1
    if num_windows <= 0:
        raise ValueError("Window size must be smaller than the length of the noisy data.")

    # Create sliding windows using array slicing
    windows = np.lib.stride_tricks.sliding_window_view(noisy_data, window_size)

    return windows

# folder_path = "../experimental_data/4um_031224/stream_00004.mat"
folder_path = "../experimental_data/7um_040224/stream_00008.mat"
# file_list = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
#
# total_time = 0

data_out, freq_labels, sample_freq = load_data(folder_path)

# Display results
# print(f"Number of channels loaded: {len(data_out)}")
# print(f"Sample frequency: {sample_freq}")
# for i, (channel_data, freq) in enumerate(zip(data_out, freq_labels)):
#     print(f"Channel {i+1}: Length = {len(channel_data)}, Frequency = {freq}")

model_path = './models/Custom_TF_stream_30_7e-4.pkl'
scaler_x_path = './scalers/Custom_TF_stream_30_7e-4_X.pkl'
scaler_y_path = './scalers/Custom_TF_stream_30_7e-4_y.pkl'
window_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

noisy_data = np.array(data_out[1])  # Replace with the desired channel index
print(f"noisy data shape: {noisy_data.shape}")
# Remove DC Bias and detrend
noisy_data_demeaned = noisy_data - np.mean(noisy_data)
noisy_data = signal.detrend(noisy_data_demeaned)

# # test with synthetic dataset
# noisy_data_path = 'data/noiselevel1e-4/noisy/'
# noisy_data = read_and_stack_csv_files(noisy_data_path).flatten()
# print(f"noisy data shape: {noisy_data.shape}")

# Generate sliding window dataset
X_test = generate_sliding_window(noisy_data, window_size)
print(f"windows_total shape: {X_test.shape}")

# Apply the neural network to denoise
denoised_signal = apply_neural_network_denoising(X_test, model_path, scaler_x_path, scaler_y_path, device="cpu")
delayed_denoised_signal = apply_delay(denoised_signal, window_size)

# Plot denoised validation samples
validation_plot_real(noisy_data, delayed_denoised_signal)

