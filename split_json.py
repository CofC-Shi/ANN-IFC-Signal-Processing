import os
import json

# File paths
input_file = './processed_signals/processed_signals.json'
output_dir = './processed_signals_by_noise_level'
os.makedirs(output_dir, exist_ok=True)

# Split JSON by noise level
with open(input_file, 'r') as f:
    processed_signals = json.load(f)

for noise_level, data in processed_signals.items():
    output_file = os.path.join(output_dir, f'processed_signals_{noise_level}.json')
    with open(output_file, 'w') as out_f:
        json.dump({noise_level: data}, out_f, indent=4)

print(f"Processed signals split into individual files in {output_dir}")
