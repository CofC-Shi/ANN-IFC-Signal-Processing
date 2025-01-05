# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import padasip as pa
import pickle
import json
import scipy.signal as ss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import DataLoader
from utils import TimeSeriesDataset, generate_sliding_window_dataset_TSF, read_and_stack_csv_files,apply_neural_network_denoising
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


def af_method(signal, delay, n_weights, mu):
    # Define Filter
    f = pa.filters.FilterNLMS(n=n_weights, mu=mu, w='zeros')
    # Delay and format input
    x = np.hstack((signal[:-delay - 1], np.zeros(delay)))
    x = pa.input_from_history(x, n_weights)
    # filter
    y, e, w = f.run(signal[n_weights:], x)

    return y, e


def snr_calc(clean_signal, noisy_signal):
    assert noisy_signal.shape == clean_signal.shape
    # Calculate signal and noise power
    signal_power = np.sum(clean_signal ** 2) / len(clean_signal)
    noise_power = np.sum((noisy_signal - clean_signal) ** 2) / len(clean_signal)

    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

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

# Load Data

## Clean Data



# Create Results structure
results = {
    '2_2e-5': {},
    '5e-5': {},
    '1e-4': {},
    '2e-4': {},
    '3e-4': {},
    '5e-4': {},
    '7e-4': {}
}

## Clean and Noisy Data
### 2.2e-5
results['2_2e-5']['clean'] = read_and_stack_csv_files('data/noiselevel2_2e-5/clean/')
results['2_2e-5']['raw'] = read_and_stack_csv_files('data/noiselevel2_2e-5/noisy/')
## 5e-5
results['5e-5']['clean'] = read_and_stack_csv_files('data/noiselevel5e-5/clean/')
results['5e-5']['raw'] = read_and_stack_csv_files('data/noiselevel5e-5/noisy/')
### 1e-4
results['1e-4']['clean'] = read_and_stack_csv_files('data/noiselevel1e-4/clean/')
results['1e-4']['raw'] = read_and_stack_csv_files('data/noiselevel1e-4/noisy/')
### 2e-4
results['2e-4']['clean'] = read_and_stack_csv_files('data/noiselevel2e-4/clean/')
results['2e-4']['raw'] = read_and_stack_csv_files('data/noiselevel2e-4/noisy/')
### 3e-4
results['3e-4']['clean'] = read_and_stack_csv_files('data/noiselevel3e-4/clean/')
results['3e-4']['raw'] = read_and_stack_csv_files('data/noiselevel3e-4/noisy/')
### 5e-4
results['5e-4']['clean'] = read_and_stack_csv_files('data/noiselevel5e-4/clean/')
results['5e-4']['raw'] = read_and_stack_csv_files('data/noiselevel5e-4/noisy/')
### 7e-4
results['7e-4']['clean'] = read_and_stack_csv_files('data/noiselevel7e-4/clean/')
results['7e-4']['raw'] = read_and_stack_csv_files('data/noiselevel7e-4/noisy/')

filtering_methods = ['nn', 'tf', 'af', 'sg']

model_path = './models/Custom_TF_stream_30_7e-4.pkl'
scaler_x_path = './scalers/Custom_TF_stream_30_7e-4_X.pkl'
scaler_y_path = './scalers/Custom_TF_stream_30_7e-4_y.pkl'
window_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

i = 0

print(f"Available noise levels: {list(results.keys())}")
for noise_level, data in results.items():
    print(f"{noise_level}: {data.keys()}")

for noise_level in results:
    clean_data = results[noise_level]['clean']
    noisy_data = results[noise_level]['raw']

    # Generate sliding window dataset
    windows_total, next_values_total = generate_sliding_window_dataset_TSF(clean_data, noisy_data, window_size)

    # Use the entire dataset for testing
    X_test = windows_total
    y_test = next_values_total


    # Apply neural network
    nn = apply_neural_network_denoising(X_test, model_path, scaler_x_path, scaler_y_path, device)


    # Filtering

    ## TF
    noisy_data = noisy_data.flatten()
    fs = 7196
    Q = 5
    ### Notch filtering of 60 hz
    b0, a0 = ss.iirnotch(60, Q, fs)
    tf = ss.filtfilt(b0, a0, noisy_data.T)
    ### Notch Filtering of 120 Hz
    b1, a1 = ss.iirnotch(120, Q, fs)
    tf = ss.filtfilt(b1, a1, tf)
    ### Lowpass filtering
    b2, a2 = ss.butter(4, 1000, btype='lowpass', output='ba', fs=fs)
    tf = ss.filtfilt(b2, a2, tf)

    ## AF
    delay = 50
    n_weights = 300
    mu = 0.5
    ### Apply function
    af, err = af_method(noisy_data, delay, n_weights, mu)

    ## SG
    window_l = 10
    polyorder = 3
    ### Apply Function
    sg = ss.savgol_filter(noisy_data, window_l, polyorder)

    ## Write results
    results[noise_level]['nn'] = nn
    results[noise_level]['tf'] = tf
    results[noise_level]['af'] = af
    results[noise_level]['sg'] = sg

    i += 1

# Scoring
for noise_level in results:
    print(noise_level)
    scores = {
        'snr': {
            'total': {},
            'event': {}
        },
        'mse': {
            'total': {},
            'event': {}
        },
        'mae': {
            'total': {},
            'event': {}
        },
        'r2': {
            'total': {},
            'event': {}
        },
    }

    ## Base SNRs
    scores['snr']['total']['raw'] = snr_calc(results[noise_level]['clean'], results[noise_level]['raw'])
    # event_snr = []
    # events = detect_event_regions(results[noise_level]['clean'], 5e-6)
    # scores['snr']['event']['raw'] = calculate_event_snr(results[noise_level]['clean'], results[noise_level]['raw'], events)

    event_snr = []

    # Process each file
    for (clean, sig) in zip(results[noise_level]['clean'], results[noise_level]['raw']):
        # Detect events
        events = detect_event_regions(clean, threshold=5e-6)

        if events:
            # Calculate SNR using identified events
            snr_value = calculate_event_snr(clean, sig, events)
            event_snr.append(snr_value)
        else:
            event_snr.append(np.nan)  # Append NaN if no events are found

    # Final SNR
    SNR_total = np.nanmean(event_snr)  # Use nanmean to skip NaNs if no events exist
    print(f"SNR total for {noise_level}: {SNR_total}")
    scores['snr']['event']['raw'] = SNR_total

    ## Method Metrics
    clean_data = results[noise_level]['clean']
    for method in filtering_methods:

        clean = clean_data
        clean = clean.flatten()

        ### Overcome delays
        if method == 'nn':
            clean = clean[window_size:]
        if method == 'af':
            clean = clean[n_weights:]

        filtered_data = results[noise_level][method]

        if clean.shape[0] != filtered_data.shape[0]:
            min_length = min(clean.shape[0], filtered_data.shape[0])
            clean = clean[:min_length]
            filtered_data = filtered_data[:min_length]

        print(method, clean.shape, filtered_data.shape)

        scores['snr']['total'][method] = snr_calc(clean, filtered_data)
        scores['mse']['total'][method] = mean_squared_error(clean, filtered_data)
        scores['mae']['total'][method] = mean_absolute_error(clean, filtered_data)
        scores['r2']['total'][method] = r2_score(clean, filtered_data)

        event_snr = []
        event_mse = []
        event_mae = []
        event_r2 = []
        events = detect_event_regions(clean, 5e-6)
        for event in events:
            event_start, event_end = event[0], event[1]

            event_snr.append(snr_calc(clean[event_start:event_end], filtered_data[event_start:event_end]))
            event_mse.append(mean_squared_error(clean[event_start:event_end], filtered_data[event_start:event_end]))
            event_mae.append(mean_absolute_error(clean[event_start:event_end], filtered_data[event_start:event_end]))
            event_r2.append(r2_score(clean[event_start:event_end], filtered_data[event_start:event_end]))

        scores['snr']['event'][method] = np.mean(event_snr)
        scores['mse']['event'][method] = np.mean(event_mse)
        scores['mae']['event'][method] = np.mean(event_mae)
        scores['r2']['event'][method] = np.mean(event_r2)

    with open(f'results/{noise_level}_results.json', 'w') as fp:
        json.dump(scores, fp, indent=4)

    print(f"Completed processing for noise level {noise_level}")

    # Select a noise level for this plot.
    nl = '7e-4'

    colors = {
        'raw': '#9BBB59',
        'nn': '#4f81bd',
        'tf': '#c0504d',
        'sg': '#8064a2'
    }

    plt.rcParams.update({"font.family": "Arial", })

    # Plotting
    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)
            # ax.xaxis.set_major_locator(ticker.NullLocator())
            # ax.yaxis.set_major_locator(ticker.NullLocator())


    fig = plt.figure(layout='constrained')
    fig.set_size_inches(14, 7)
    subfigs = fig.subfigures(2, 1)

    ax1 = subfigs[0].subplots(1, 1)
    ax_bottom = subfigs[1].subplots(1, 3, sharey=True)
    ax2 = ax_bottom[0]
    ax3 = ax_bottom[1]
    ax4 = ax_bottom[2]

    # ax1
    # ax1.set_title('Noise level: 1e-4')
    ax1.plot(results[nl]['raw'][window_size:], label='Noisy', color=colors['raw'])
    ax1.plot(clean_data[window_size:], label='Noiseless', color='#F79646', linewidth=3)
    ax1.set_xlim([60000, 77500])
    ax1.legend(loc='upper right', fontsize='large')
    ax1.set_title(f'Synthetically Generated Data with Noise Level {nl}', fontsize=16)
    ax1.set_ylabel('Amplitude (V)', fontsize=16)
    ax1.set_xlabel('Samples', fontsize=16)

    # ax2
    ax2.plot(clean_data[window_size:], label='Actual', color=colors['raw'], linewidth=3)
    # ax2.plot(results[nl]['af'], label='af', linestyle=':', color='blue')
    ax2.plot(results[nl]['sg'][window_size:], label='sg', linestyle='-.', color=colors['sg'])
    ax2.plot(results[nl]['nn'], label='nn', linestyle='--', color=colors['nn'])
    ax2.plot(results[nl]['tf'].T[window_size:], label='tf', linestyle=':', color=colors['tf'])
    ax2.set_xlim([64850, 65000])
    # ax2.set_ylim([-0.0006, 0.00035])
    ax2.legend(loc='lower right', fontsize='large')
    ax2.set_ylabel('Amplitude (V)', fontsize=16)
    ax2.set_xlabel('Samples', fontsize=16)

    # ax3
    ax3.plot(clean_data[window_size:], label='Actual', color=colors['raw'], linewidth=3)
    # ax3.plot(results[nl]['af'], label='af', linestyle=':', color='blue')
    ax3.plot(results[nl]['sg'][window_size:], label='sg', linestyle='-.', color=colors['sg'])
    ax3.plot(results[nl]['nn'], label='nn', linestyle='--', color=colors['nn'])
    ax3.plot(results[nl]['tf'].T[window_size:], label='tf', linestyle=':', color=colors['tf'])
    ax3.set_xlim([67540, 67650])
    # ax3.set_ylim([-0.0006, 0.0004])
    ax3.legend(loc='lower right', fontsize='large')
    ax3.set_xlabel('Samples', fontsize=16)

    # ax4
    ax4.plot(clean_data[window_size:], label='Actual', color=colors['raw'], linewidth=3)
    # ax4.plot(results[nl]['af'], label='af', linestyle=':', color='blue')
    ax4.plot(results[nl]['sg'][window_size:], label='sg', linestyle='-.', color=colors['sg'])
    ax4.plot(results[nl]['nn'], label='nn', linestyle='--', color=colors['nn'])
    ax4.plot(results[nl]['tf'].T[window_size:], label='tf', linestyle=':', color=colors['tf'])
    ax4.set_xlim([72875, 73000])
    ax4.set_ylim([-0.00065, 0.0004])
    ax4.legend(loc='lower right', fontsize='large')
    ax4.set_xlabel('Samples', fontsize=16)

    # format_axes(fig)
    plt.savefig(f"./results/plot_noise_{nl}.png", bbox_inches='tight')

    plt.show()
