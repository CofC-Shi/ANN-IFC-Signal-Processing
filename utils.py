import numpy as np
import pandas as pd
import glob
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.signal import spectrogram
import librosa
import scipy.io
from scipy.io import loadmat
from torch.utils.data import DataLoader
import torch


class TimeSeriesDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def read_and_stack_csv_files(data_path):
    """
        Reads and stacks all CSV files from a given directory into a single NumPy array.
    """
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_path}")
        return np.array([])
    data_list = [pd.read_csv(file, header=None, skiprows=1).values.flatten() for file in csv_files]
    stacked_data = np.vstack(data_list)
    return stacked_data


def generate_sliding_window_dataset(clean_data, noisy_data, window_size):
    """
    Generates sliding windows from noisy data and the corresponding clean windows.
    """
    num_streams, stream_length = clean_data.shape
    assert noisy_data.shape == clean_data.shape, "Noisy and clean data must have the same shape."

    windows_noisy = []
    windows_clean = []

    for i in range(num_streams):
        for j in range(stream_length - window_size + 1):
            windows_noisy.append(noisy_data[i, j:j + window_size])
            windows_clean.append(clean_data[i, j:j + window_size])

    windows_noisy = np.array(windows_noisy, dtype=np.float32)
    windows_clean = np.array(windows_clean, dtype=np.float32)

    return windows_noisy, windows_clean

def generate_sliding_window_dataset_TSF(clean_data, noisy_data, window_size):
    """
       Generates sliding windows from noisy data and corresponding next values from clean data.
    """
    num_streams, stream_length = clean_data.shape
    assert noisy_data.shape == clean_data.shape
    windows_total = []
    next_values_total = []

    for i in range(num_streams):
        for j in range(stream_length - window_size):
            windows_total.append(noisy_data[i, j:j + window_size])
            next_values_total.append(clean_data[i, j + window_size])

    windows_total = np.array(windows_total, dtype=np.float32)
    next_values_total = np.array(next_values_total, dtype=np.float32).reshape(-1, 1)

    return windows_total, next_values_total

def reconstruct_signal_from_windows(windows, original_length, window_size):
    """
    Reconstructs the original signal from overlapping windows using averaging.
    """
    reconstructed_signal = np.zeros(original_length)
    weight_array = np.zeros(original_length)

    num_windows = len(windows)
    for i in range(num_windows):
        start_idx = i
        end_idx = start_idx + window_size
        reconstructed_signal[start_idx:end_idx] += windows[i]
        weight_array[start_idx:end_idx] += 1

    # Avoid division by zero and average overlapping contributions
    reconstructed_signal /= np.maximum(weight_array, 1)

    return reconstructed_signal

def validation_plot(X_test, y_test, y_pred, num_samples=1, data_length=3000):

    plt.figure(figsize=(12, num_samples * 6))
    for i in range(num_samples):
        idx = np.random.randint(0, len(X_test) - data_length)

        # Plot noisy data
        plt.subplot(num_samples * 3, 1, 3 * i + 1)
        plt.plot(range(data_length), X_test[idx:idx + data_length], label='Noisy Data', color='gray')
        plt.title(f"Sample {i + 1}: Noisy Data")
        plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        plt.legend()

        # Plot clean data
        plt.subplot(num_samples * 3, 1, 3 * i + 2)
        plt.plot(range(data_length), y_test[idx:idx + data_length], label='Clean Data', color='blue')
        plt.title(f"Sample {i + 1}: Clean Data")
        plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        plt.legend()

        # Plot denoised data
        plt.subplot(num_samples * 3, 1, 3 * i + 3)
        plt.plot(range(data_length), y_pred[idx:idx + data_length], label='Denoised Data', color='orange')
        plt.title(f"Sample {i + 1}: Denoised Data")
        plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    plt.savefig('./results/validation_plot.png')  # Save the plot as an image
    plt.close()

def apply_delay(signal, delay_samples):

  delayed_signal = np.zeros_like(signal)
  delayed_signal[delay_samples:] = signal[:-delay_samples]

  return delayed_signal

def validation_plot_real(X_test, y_pred, num_samples=1):

    plt.figure(figsize=(8, num_samples * 4))
    for i in range(num_samples):
        # idx = np.random.randint(0, len(X_test) - data_length)
        idx = 0
        data_length = len(y_pred)

        # Plot noisy data
        plt.subplot(num_samples * 2, 1, 2 * i + 1)
        plt.plot(range(data_length), X_test[idx:idx + data_length], label='Noisy Data',  color='gray')
        # plt.title(f"Noisy Data")
        # plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        plt.ylim([-0.0007, 0.0003])
        plt.legend(loc='lower left')

        # Plot denoised data
        plt.subplot(num_samples * 2, 1, 2 * i + 2)
        plt.plot(range(data_length), y_pred[idx:idx + data_length], label='Denoised Data', color='orange')
        # plt.title(f"Denoised Data")
        plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        plt.ylim([-0.0007, 0.0003])
        plt.legend(loc='lower left')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    plt.savefig('./results/validation_plot_real.png')  # Save the plot as an image
    plt.close()

def save_scalers(scaler_X, scaler_y, model_name):
    os.makedirs('scalers', exist_ok=True)
    with open(f'scalers/{model_name}_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(f'scalers/{model_name}_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    print("Scalers saved successfully.")

def load_data(data_address):
    """
    Loads and formats data from a .mat file.
    Extracts all six channels from the 'dev18244.demods' structure.
    """
    # Load the .mat file
    data_struct = loadmat(data_address, struct_as_record=False, squeeze_me=True)

    # Access 'dev18244.demods'
    demods = data_struct['dev18244'].demods

    # Preallocate lists for output
    data_out = []
    freq_labels = []
    phase = []
    sample_freq = None

    # Loop through each channel (demodulator)
    for demod in demods:
        # Extract x, y, and phase components of the sample
        x = np.array(demod.sample.x)
        y = np.array(demod.sample.y)

        # Compute the magnitude (r)
        r = np.sqrt(x**2 + y**2)

        # Append the magnitude to the data output
        data_out.append(r)

        # Extract frequency
        if isinstance(demod.freq, np.ndarray) or hasattr(demod.freq, 'value'):
            freq = demod.freq.value if hasattr(demod.freq, 'value') else demod.freq
            freq_labels.append(freq)

        # Extract sampling frequency (only once)
        if sample_freq is None and hasattr(demod.rate, 'value'):
            sample_freq = demod.rate.value

    return data_out, freq_labels, sample_freq

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
            batch_X = batch_X.float()
            batch_X = batch_X.to(device)
            outputs = model(batch_X).cpu().numpy()
            denoised_signal.append(outputs)

    denoised_signal = np.concatenate(denoised_signal, axis=0)
    denoised_signal = scaler_y.inverse_transform(denoised_signal)

    return denoised_signal.flatten()


