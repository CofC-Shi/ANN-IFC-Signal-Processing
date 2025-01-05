# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import padasip as pa
import pickle
import json
import scipy.signal as ss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rms(array):
    """
       Computes the Root Mean Square (RMS) of an array.

       Args:
           array (np.ndarray): Input array.

       Returns:
           float: RMS value of the array.
   """
    return np.sqrt(np.mean(np.square(array)))


def detect_events(signal, threshold):
    """
        Detects events in a signal based on crossing a threshold.

        Args:
            signal (np.ndarray): 1D array representing the signal.
            threshold (float): Value for detecting positive and negative peaks.

        Returns:
            list: List of tuples (start_idx, end_idx) for each detected event.
    """
    events = []
    in_pos_peak = False
    in_event = False
    start_idx = 0
    for i, value in enumerate(signal):
        if value < -threshold and not in_event:
            in_event = True
            start_idx = i
        elif value > threshold:
            in_pos_peak = True
        elif value < threshold:
            if in_pos_peak:
                end_idx = i
                events.append((start_idx, end_idx))
                in_pos_peak = False
                in_event = False
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
    noise = noisy_signal - clean_signal
    snr = 20 * np.log10(rms(clean_signal) / rms(noise))
    return snr


def snr_calc_updated(clean_signal, noisy_signal):
    S = np.max(clean_signal)
    N = rms(noisy_signal - clean_signal)
    print(N)
    return 20 * np.log10(S / N)


def generate_sliding_window_dataset(clean_data, noisy_data, window_size):
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


# Load Data

## Clean Data
clean_data = pd.read_csv('data/noiselevel1e-4/clean/clean_1.csv', header=None, skiprows=1).values
### Find Events
events = detect_events(clean_data, 5e-6)

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

## Noisy Data
### 2.2e-5
results['2_2e-5']['raw'] = pd.read_csv('data/noiselevel2_2e-5/noisy/noisy_1.csv', header=None, skiprows=1).values
## 5e-5
results['5e-5']['raw'] = pd.read_csv('data/noiselevel5e-5/noisy/noisy_1.csv', header=None, skiprows=1).values
### 1e-4
results['1e-4']['raw'] = pd.read_csv('data/noiselevel1e-4/noisy/noisy_1.csv', header=None, skiprows=1).values
### 2e-4
results['2e-4']['raw'] = pd.read_csv('data/noiselevel2e-4/noisy/noisy_1.csv', header=None,
                                     skiprows=1).values
### 3e-4
results['3e-4']['raw'] = pd.read_csv('data/noiselevel3e-4/noisy/noisy_1.csv', header=None, skiprows=1).values
### 5e-4
results['5e-4']['raw'] = pd.read_csv('data/noiselevel5e-4/noisy/noisy_1.csv', header=None, skiprows=1).values
### 7e-4
results['7e-4']['raw'] = pd.read_csv('data/noiselevel7e-4/noisy/noisy_1.csv', header=None, skiprows=1).values

filtering_methods = ['nn', 'tf', 'af', 'sg']

model_name = '071624_streams30_rate5_1e-4_256_64'
window_size = 32
### Load Model and Scalers
with open(f'models/{model_name}.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'scalers/{model_name}_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open(f'scalers/{model_name}_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

i = 0

print(f"Available noise levels: {list(results.keys())}")
for noise_level, data in results.items():
    print(f"{noise_level}: {data.keys()}")

for noise_level in results:
    noisy_data = results[noise_level]['raw']
    noisy_data = noisy_data.reshape((len(noisy_data), 1))

    # Filtering

    ## NN
    ### Apply Sliding window
    windows, nvs = generate_sliding_window_dataset(clean_data.T, noisy_data.T, window_size)
    X = scaler_X.transform(windows)
    ### Prediction
    nn = model.predict(X)
    nn = scaler_y.inverse_transform(nn.reshape(-1, 1))

    ## TF
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
    af, err = af_method(noisy_data[:, 0], delay, n_weights, mu)

    ## SG
    window_l = 10
    polyorder = 3
    ### Apply Function
    sg = ss.savgol_filter(noisy_data[:, 0], window_l, polyorder)

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
    scores['snr']['total']['raw'] = snr_calc_updated(clean_data, results[noise_level]['raw'])
    event_snr = []
    events = detect_events(clean_data, 1e-6)
    for event in events:
        event_start, event_end = event[0], event[1]
        event_snr.append(
            snr_calc(clean_data[event_start:event_end], results[noise_level]['raw'][event_start:event_end]))
    scores['snr']['event']['raw'] = np.mean(event_snr)

    ## Method Metrics
    for method in filtering_methods:

        clean = clean_data

        ### Overcome delays
        if method == 'nn':
            clean = clean[window_size:]
        if method == 'af':
            clean = clean[n_weights:]

        filtered_data = results[noise_level][method]

        if filtered_data.shape != clean.shape:
            filtered_data = np.reshape(filtered_data, clean.shape)

        print(method, clean.shape, filtered_data.shape)

        scores['snr']['total'][method] = snr_calc_updated(clean, filtered_data)
        scores['mse']['total'][method] = mean_squared_error(clean, filtered_data)
        scores['mae']['total'][method] = mean_absolute_error(clean, filtered_data)
        scores['r2']['total'][method] = r2_score(clean, filtered_data)

        event_snr = []
        event_mse = []
        event_mae = []
        event_r2 = []
        events = detect_events(clean, 1e-6)
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
        #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)
        #ax.xaxis.set_major_locator(ticker.NullLocator())
        #ax.yaxis.set_major_locator(ticker.NullLocator())


fig = plt.figure(layout='constrained')
fig.set_size_inches(14, 7)
subfigs = fig.subfigures(2, 1)

ax1 = subfigs[0].subplots(1, 1)
ax_bottom = subfigs[1].subplots(1, 3, sharey=True)
ax2 = ax_bottom[0]
ax3 = ax_bottom[1]
ax4 = ax_bottom[2]

# ax1
#ax1.set_title('Noise level: 1e-4')
ax1.plot(results[nl]['raw'][window_size:], label='Noisy', color=colors['raw'])
ax1.plot(clean_data[window_size:], label='Noiseless', color='#F79646', linewidth=3)
ax1.set_xlim([60000, 77500])
ax1.legend(loc='upper right', fontsize='large')
ax1.set_title(f'Synthetically Generated Data with Noise Level {nl}', fontsize=16)
ax1.set_ylabel('Amplitude (V)', fontsize=16)
ax1.set_xlabel('Samples', fontsize=16)

# ax2
ax2.plot(clean_data[window_size:], label='Actual', color=colors['raw'], linewidth=3)
#ax2.plot(results[nl]['af'], label='af', linestyle=':', color='blue')
ax2.plot(results[nl]['sg'][window_size:], label='sg', linestyle='-.', color=colors['sg'])
ax2.plot(results[nl]['nn'], label='nn', linestyle='--', color=colors['nn'])
ax2.plot(results[nl]['tf'].T[window_size:], label='tf', linestyle=':', color=colors['tf'])
ax2.set_xlim([64850, 65000])
#ax2.set_ylim([-0.0006, 0.00035])
ax2.legend(loc='lower right', fontsize='large')
ax2.set_ylabel('Amplitude (V)', fontsize=16)
ax2.set_xlabel('Samples', fontsize=16)

# ax3
ax3.plot(clean_data[window_size:], label='Actual', color=colors['raw'], linewidth=3)
#ax3.plot(results[nl]['af'], label='af', linestyle=':', color='blue')
ax3.plot(results[nl]['sg'][window_size:], label='sg', linestyle='-.', color=colors['sg'])
ax3.plot(results[nl]['nn'], label='nn', linestyle='--', color=colors['nn'])
ax3.plot(results[nl]['tf'].T[window_size:], label='tf', linestyle=':', color=colors['tf'])
ax3.set_xlim([67540, 67650])
#ax3.set_ylim([-0.0006, 0.0004])
ax3.legend(loc='lower right', fontsize='large')
ax3.set_xlabel('Samples', fontsize=16)

# ax4
ax4.plot(clean_data[window_size:], label='Actual', color=colors['raw'], linewidth=3)
#ax4.plot(results[nl]['af'], label='af', linestyle=':', color='blue')
ax4.plot(results[nl]['sg'][window_size:], label='sg', linestyle='-.', color=colors['sg'])
ax4.plot(results[nl]['nn'], label='nn', linestyle='--', color=colors['nn'])
ax4.plot(results[nl]['tf'].T[window_size:], label='tf', linestyle=':', color=colors['tf'])
ax4.set_xlim([72875, 73000])
ax4.set_ylim([-0.00065, 0.0004])
ax4.legend(loc='lower right', fontsize='large')
ax4.set_xlabel('Samples', fontsize=16)

#format_axes(fig)
plt.savefig(f"./results/plot_noise_{nl}.png", bbox_inches='tight')

plt.show()
