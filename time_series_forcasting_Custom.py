# Import Libraries
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import torch
import yaml
from model import CustomModelTSF
from torch.utils.data import Dataset, DataLoader
from trainer import train_model
from utils import (
    TimeSeriesDataset,
    read_and_stack_csv_files,
    generate_sliding_window_dataset_TSF,
    reconstruct_signal_from_windows,
    validation_plot,
    save_scalers,
    apply_neural_network_denoising,
)

matplotlib.use('Qt5Agg')
def main():
    ###################################################
    # set up argparse and device housekeeping
    ###################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device is: {device}")

    # Data Loading/Preprocessing

    window_size = args.window_size
    test_size = args.test_size
    model_name = f'Custom_TF_stream_30_7e-4_symmetric'

    # Paths to directories containing clean and noisy data
    clean_data_path = 'data/noiselevel7e-4_30streams_symmetric/clean/'
    noisy_data_path = 'data/noiselevel7e-4_30streams_symmetric/noisy/'

    # clean_data_path = 'data/noiselevel7e-4/clean/'
    # noisy_data_path = 'data/noiselevel7e-4/noisy/'

    print(f'clean path: {clean_data_path}')

    # Read and stack all CSV files for clean and noisy data
    clean_data = read_and_stack_csv_files(clean_data_path)
    noisy_data = read_and_stack_csv_files(noisy_data_path)

    print(f"Clean data shape: {clean_data.shape}, Noisy data shape: {noisy_data.shape}")

    if clean_data.size == 0 or noisy_data.size == 0:
        raise ValueError("Failed to read data from CSV files. Please check the file paths and contents.")
    if clean_data.shape != noisy_data.shape:
        raise ValueError("Clean data and noisy data must have the same shape.")

    windows_total, next_values_total = generate_sliding_window_dataset_TSF(clean_data, noisy_data, window_size)
    del clean_data, noisy_data

    print(f"Shape of windows_total: {windows_total.shape}, Shape of next_values_total: {next_values_total.shape}")

    ###################################################
    # Model Training
    ###################################################
    X_train, X_test, y_train, y_test = train_test_split(windows_total, next_values_total, test_size=test_size,
                                                        shuffle=False)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Shuffle training data
    train_indices = np.arange(len(X_train))
    np.random.shuffle(train_indices)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    ###################################################
    # Training Loop
    ###################################################
    if args.train_model:

        # Initialize scalers
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))

        save_scalers(scaler_X, scaler_y, model_name)

        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test.reshape(-1, 1))

        # load data in batches
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Keep original test data for evaluation
        X_test_original = scaler_X.inverse_transform(X_test)
        y_test_original = scaler_y.inverse_transform(y_test)
        y_train_original = scaler_y.inverse_transform(y_train)

        ###################################################
        # Model Selection and Initialization
        ###################################################
        model = CustomModelTSF(window_size).to(device)

        ###################################################
        # Training Setup
        ###################################################
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        num_epochs = args.num_epochs

        print("Starting training...")
        # replace X_train_tensor, y_train_tensor with Dataloader for batch training
        model, loss_history, train_time = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=num_epochs,
            device=device,
            train_loader=train_loader,
            model_name=model_name,
        )
        print("Training complete!")

    ###################################################
    # Model Evaluation
    ###################################################
        # Evaluate the model on training data
        model.eval()  # Set model to evaluation mode
        all_y_pred_train = []
        all_y_pred_test = []
        with torch.no_grad():
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).cpu().numpy()
                torch.onnx.export(model, batch_X, "custom_model.onnx")
                all_y_pred_train.append(outputs)

            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).cpu().numpy()
                all_y_pred_test.append(outputs)

            # Concatenate all batches
            y_pred_train = np.concatenate(all_y_pred_train, axis=0)
            y_pred_train_original = scaler_y.inverse_transform(y_pred_train)

            y_pred_test = np.concatenate(all_y_pred_test, axis=0)
            y_pred_test_original = scaler_y.inverse_transform(y_pred_test)

        # Save Model
        with open(f'./models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Plotting Loss Curve
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'results/{args.model}.png')
        print("Model and results saved successfully.")

        # Compute metrics
        mse_train = mean_squared_error(y_train_original, y_pred_train_original)  # Training MSE
        mse_test = mean_squared_error(y_test_original, y_pred_test_original)  # Testing MSE

        # Print results
        print(f"Training Time: {train_time} s")
        print(f"Training MSE: {mse_train}")
        print(f"Test MSE: {mse_test}")

        # Reconstruct signals
        original_length = y_test.shape[0] + window_size - 1  # Calculate the original signal length
        noisy_signal_reconstructed = reconstruct_signal_from_windows(X_test_original, original_length, window_size)

        # Plot denoised validation samples
        validation_plot(noisy_signal_reconstructed, y_test_original, y_pred_test_original)

    else:
        # Define file paths
        model_path = './models/Custom_TF_stream_30_7e-4.pkl'
        scaler_x_path = './scalers/Custom_TF_stream_30_7e-4_X.pkl'
        scaler_y_path = './scalers/Custom_TF_stream_30_7e-4_y.pkl'

        y_pred_test_original = apply_neural_network_denoising(X_test, model_path, scaler_x_path, scaler_y_path, device="cpu")

        # Compute metrics
        mse_test = mean_squared_error(y_test, y_pred_test_original)
        mae_test = mean_absolute_error(y_test, y_pred_test_original)
        r2_test = r2_score(y_test, y_pred_test_original)

        print(f"Test MSE: {mse_test}")
        print(f"Test MAE: {mae_test}")
        print(f"Test R2: {r2_test}")

        original_length = y_test.shape[0] + window_size - 1  # Calculate the original signal length
        noisy_signal_reconstructed = reconstruct_signal_from_windows(X_test, original_length, window_size)

        # Plot denoised validation samples
        validation_plot(noisy_signal_reconstructed, y_test, y_pred_test_original)


if __name__ == "__main__":
    main()
