# ANN-Based Signal Processing for Impedance Flow Cytometry (IFC)
## Overview
Impedance flow cytometry (IFC) is a label-free technique for characterizing micron-scale particles, offering advantages such as non-invasive analysis and high-throughput capabilities. However, the inherent noise and variability in impedance signals pose significant challenges to reliable particle characterization.

This project introduces a novel signal processing framework leveraging an Artificial Neural Network (ANN) for denoising raw time-domain impedance signals. By integrating time-series forecasting with ANN-based filtering, the method effectively reduces noise while preserving key signal features critical for particle analysis. The denoised signals are processed to extract features that serve as inputs for machine learning classifiers to distinguish between different particle populations.

Experimental evaluations show that this ANN-based approach outperforms traditional filtering techniques, such as Savitzky-Golay and Notch filters, across varying noise levels. Using Random Forest and k-Nearest Neighbors models, the system achieves an area-under-the-curve (AUC) value of 0.96 and an overall accuracy of 92%. These results demonstrate the potential of combining advanced signal processing with machine learning to enhance the precision and robustness of particle characterization in label-free IFC systems.

## Features
+ Signal Denoising with ANN: Leverages an ANN-based framework for noise reduction in raw time-domain impedance signals.

+ Time-Series Forecasting Integration: Incorporates a time-series forecasting model to improve denoising performance.

+ Feature Extraction for Classification: Processes denoised signals to extract meaningful features for particle classification.

+ Machine Learning Classifiers: Supports Random Forest and k-Nearest Neighbors classifiers for particle population distinction.

+ Noise Robustness: Demonstrates improved performance across different levels of periodic, white, and pink noise.

## Repository Structure
```
├── .git
├── .idea
├── Config
│   └── config.yaml
├── data
│   ├── noiselevel1e-4
│   ├── noiselevel1e-4_30streams
│   └── ...
├── models
│   ├── Custom_TF_stream_30_1e-4.pkl
|   ├── Custom_TF_stream_30_2e-4.pkl
|   ├── Custom_TF_stream_30_5e-4.pkl
│   └── ...
├── processed_signals_by_noise_level
│   ├── processed_signals_1e-4.json
│   ├── processed_signals_2_2e-5.json
│   └── ...
├── results
│   ├── 1e-4_results.json
│   ├── 2_2e-5_results.json
│   └── ...
├── scalers
│   ├── Custom_TF_stream_30_1e-4_X.pkl
│   ├── Custom_TF_stream_30_1e-4_y.pkl
│   └── ...
├── ANN_proj.yml
├── batch_real_data_process.py
├── beads_classification.py
├── calculate_SNR.py
├── dataset_visualization.py
├── data_generation.py
├── method_comparison.py
├── model.py
├── plot_denoised_data.py
├── ReadMe.md
├── real_data_process.py
├── time_series_forecasting_Custom.py
├── traditional_processing.py
├── trainer.py
├── trend_plot.py
├── utils.py
```
## Contact
For questions or feedback, reach out via email at shil1@cofc.edu or open an issue on GitHub.
