import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")  # Use the Qt5Agg backend
# Load results files
result_files = [
    "./results/2_2e-5_results.json",
    "./results/5e-5_results.json",
    "./results/1e-4_results.json",
    "./results/2e-4_results.json",
    "./results/3e-4_results.json",
    "./results/5e-4_results.json",
    "./results/7e-4_results.json",
]

# Define methods and metrics
methods = ['nn', 'tf', 'sg']
metrics = ['snr', 'mse', 'mae', 'r2']

# Initialize data for plotting
input_event_snr = []
output_event_snr = {method: [] for method in methods}
event_mse = {method: [] for method in methods}
event_mae = {method: [] for method in methods}
event_r2 = {method: [] for method in methods}

# Extract data from files
for file in result_files:
    with open(file, 'r') as f:
        results = json.load(f)
        input_event_snr.append(results['snr']['event']['raw'])
        for method in methods:
            output_event_snr[method].append(results['snr']['event'][method])
            event_mse[method].append(results['mse']['event'][method])
            event_mae[method].append(results['mae']['event'][method])
            event_r2[method].append(results['r2']['event'][method])

# Plotting
plt.figure(figsize=(10, 7.5))

# Subplot 1: Output Event SNR vs. Input Event SNR
plt.subplot(2, 2, 1)
for method in methods:
    label = "ANN" if method == 'nn' else "Notch" if method == 'tf' else "Savitzky-Golay"
    color = 'red' if method == 'tf' else None
    plt.plot(input_event_snr, output_event_snr[method], label=label, marker='o', color=color)
plt.title("Output Event SNR vs. Input Event SNR")
plt.xlabel("Input Event SNR (dB)")
plt.ylabel("Output Event SNR (dB)")
plt.legend()
plt.grid(True)

# Subplot 2: Event MSE vs. Input Event SNR
plt.subplot(2, 2, 2)
for method in methods:
    label = "ANN" if method == 'nn' else "Notch" if method == 'tf' else "Savitzky-Golay"
    color = 'red' if method == 'tf' else None
    plt.plot(input_event_snr, event_mse[method], label=label, marker='o', color=color)
plt.title("Event MSE vs. Input Event SNR")
plt.xlabel("Input Event SNR (dB)")
plt.ylabel("Event MSE")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend()
plt.grid(True)

# Subplot 3: Event MAE vs. Input Event SNR
plt.subplot(2, 2, 3)
for method in methods:
    label = "ANN" if method == 'nn' else "Notch" if method == 'tf' else "Savitzky-Golay"
    color = 'red' if method == 'tf' else None
    plt.plot(input_event_snr, event_mae[method], label=label, marker='o', color=color)
plt.title("Event MAE vs. Input Event SNR")
plt.xlabel("Input Event SNR (dB)")
plt.ylabel("Event MAE")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend()
plt.grid(True)

# Subplot 4: Event R2 vs. Input Event SNR
plt.subplot(2, 2, 4)
for method in methods:
    label = "ANN" if method == 'nn' else "Notch" if method == 'tf' else "Savitzky-Golay"
    color = 'red' if method == 'tf' else None
    plt.plot(input_event_snr, event_r2[method], label=label, marker='o', color=color)
plt.title("Event R$^2$ vs. Input Event SNR")
plt.xlabel("Input Event SNR (dB)")
plt.ylabel("Event R$^2$")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
