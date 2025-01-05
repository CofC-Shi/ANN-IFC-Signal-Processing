import matplotlib.pyplot as plt
import numpy as np
import json
import os
import matplotlib

matplotlib.use("Qt5Agg")  # Use the Qt5Agg backend

def load_noise_level_data(noise_level, base_dir='./processed_signals_by_noise_level', limit=3000):
    file_path = os.path.join(base_dir, f'processed_signals_{noise_level}.json')
    with open(file_path, 'r') as f:
        data = json.load(f)[noise_level]
        # Flatten and limit
        limited_data = {key: np.array(value).flatten()[2000:limit] for key, value in data.items()}

    return limited_data

# Load processed signals
# Select a noise level for visualization
selected_noise_level = '1e-4'  # Adjust based on available noise levels
data = load_noise_level_data(selected_noise_level, limit=3000)

# print("data has been loaded.")
# print(f"noise level: {selected_noise_level}, data length: {len(data)}")
# print(data)

# Plot parameters
methods = ['notch', 'savgol', 'neural_network']
colors = {
    'clean': 'black',
    'noisy': 'gray',
    'notch': 'blue',
    'savgol': 'green',
    'neural_network': 'red'
}
label_mapping = {
    'notch': 'Notch',
    'savgol': 'Savitzky-Golay',  # Updated label
    'neural_network': 'ANN'  # Updated label
}
line_styles = {
    'clean': '-',
    'noisy': '--',
    'notch': '-.',
    'savgol': ':',
    'neural_network': '-'
}

# ** Full signal plot (first figure) **
clean_signal = np.array(data['clean'])
noisy_signal = np.array(data['noisy'])

plt.figure(figsize=(10, 5))
plt.plot(noisy_signal, label='Noisy', color=colors['noisy'], linestyle=line_styles['noisy'], linewidth=1.5)
plt.plot(clean_signal, label='Actual Signal', color=colors['clean'], linestyle=line_styles['clean'], linewidth=2)

# Highlight region of interest (ROI)
roi_start, roi_end = 800, 900
plt.axvspan(roi_start, roi_end, color='red', alpha=0.3, label="Selected Region")
plt.title(f"Original Signals with SNR=16 for Processing")
plt.xlabel("Samples")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./results/original_vs_noisy_{selected_noise_level}.png', dpi=300)
plt.show()

# ** Filtered outputs for selected region (second figure) **
plt.figure(figsize=(10, 5))
x_axis = np.arange(roi_start, roi_end)
plt.plot(x_axis, clean_signal[roi_start:roi_end], label='Actual Signal', color=colors['clean'], linestyle=line_styles['clean'], linewidth=2)

for method in methods:
    processed_signal = np.array(data[method])
    method_label = label_mapping[method]
    if method == 'neural_network':
        plt.plot(x_axis, processed_signal[roi_start:roi_end], label=method_label,
                 color=colors[method], linestyle=line_styles[method], linewidth=2, alpha=0.7)  # Transparency added
    else:
        plt.plot(x_axis, processed_signal[roi_start:roi_end], label=method_label,
                 color=colors[method], linestyle=line_styles[method], linewidth=2)

plt.title(f"Filtered Outputs for Selected Region")
plt.xlabel("Samples")
plt.ylabel("Amplitude (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./results/filtered_outputs_{selected_noise_level}.png', dpi=300)
plt.show()