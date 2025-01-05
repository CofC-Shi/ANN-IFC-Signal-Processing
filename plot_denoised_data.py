import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_noise_level_data(noise_level, base_dir='./processed_signals_by_noise_level', limit=3000):
    file_path = os.path.join(base_dir, f'processed_signals_{noise_level}.json')
    with open(file_path, 'r') as f:
        data = json.load(f)[noise_level]
    limited_data = {key: np.array(value[:limit]) for key, value in data.items()}
    return limited_data

# Load processed signals
# Select a noise level for visualization
selected_noise_level = '1e-4'  # Adjust based on available noise levels
data = load_noise_level_data(selected_noise_level, limit=3000)

# Plot parameters
methods = ['noisy', 'notch', 'savgol', 'neural_network']
colors = {
    'clean': 'black',
    'noisy': 'gray',
    'notch': 'blue',
    'savgol': 'green',
    'neural_network': 'red'
}
line_styles = {
    'clean': '-',
    'noisy': '--',
    'notch': '-.',
    'savgol': ':',
    'neural_network': '-'
}

clean_signal = np.array(data['clean'])

# Plot the clean signal and processed signals
plt.figure(figsize=(14, 7))
plt.plot(clean_signal, color=colors['clean'], linestyle=line_styles['clean'], label='Clean Signal', linewidth=2)

for method in methods:
    processed_signal = np.array(data[method])
    plt.plot(processed_signal, color=colors[method], linestyle=line_styles[method], label=f'{method.capitalize()}')

plt.title(f"Signal Verification - Noise Level: {selected_noise_level}", fontsize=16)
plt.xlabel("Samples", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save and display the plot
plt.savefig(f'verification_plot_{selected_noise_level}.png', dpi=300)
plt.show()
