import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("Qt5Agg")  # Use the Qt5Agg backend
# Load both datasets
data1 = pd.read_csv('./data/extracted_features_4um_03.csv')
data2 = pd.read_csv('./data/extracted_features_7um_03.csv')

# Filter for Channel 1 data only (as an example)
data1_filtered = data1[['Amplitude', 'Width time (ms)']].copy()
data2_filtered = data2[['Amplitude', 'Width time (ms)']].copy()

# Determine the minimum class size for balanced sampling
min_size = min(len(data1_filtered), len(data2_filtered))


# Sample each dataset based on the minimum class size
data1_sampled = data1_filtered.sample(n=min_size, random_state=1)
data2_sampled = data2_filtered.sample(n=min_size, random_state=1)

plt.figure(figsize=(6, 5))

# Plot the scatter points
plt.scatter(abs(data1_sampled['Amplitude']), data1_sampled['Width time (ms)'],
            color='blue', alpha=0.7, label='4μm Beads', edgecolor='none')
plt.scatter(abs(data2_sampled['Amplitude']), data2_sampled['Width time (ms)'],
            color='red', alpha=0.7, label='7μm Beads', edgecolor='none')

# Get current axes and set log scale
ax = plt.gca()
ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(4e-4, 1e-3)

# Add labels and legend
plt.xlabel('Peak Amplitude (V)')
plt.ylabel('Transient Time (ms)')
plt.legend()

# Improve layout
plt.tight_layout()

# Display the plot
plt.show()