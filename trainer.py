import time
import torch

def train_model(
    model, optimizer, criterion, num_epochs, device, scheduler=None,
    train_loader=None, X_train_tensor=None, y_train_tensor=None, model_name=None):
    """
    Generalized training function for both batch training and training on full data.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (torch.nn.Module): Loss function.
        num_epochs (int): Number of training epochs.
        device (str): Device to use ('cpu' or 'cuda').
        train_loader (DataLoader, optional): DataLoader for batch training. If None, full data tensors are used.
        X_train_tensor (torch.Tensor, optional): Input data tensor for full-data training.
        y_train_tensor (torch.Tensor, optional): Target data tensor for full-data training.

    Returns:
        model (torch.nn.Module): Trained model.
        loss_history (list): List of loss values for each epoch.
        training_time (float): Total training time in seconds.
    """
    assert (train_loader is not None) or (X_train_tensor is not None and y_train_tensor is not None), \
        "Provide either train_loader for batch training or X_train_tensor and y_train_tensor for full-data training."

    model.train()
    loss_history = []
    start_time = time.time()

    # # Early stopping variables
    # best_loss = float('inf')
    # epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        if train_loader:  # Batch training
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.unsqueeze(1) # shape [batch_size, sequence_length, input_size]
                batch_y = batch_y.unsqueeze(1)
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)

        else:  # Full-data training
            X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)

            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()

        if scheduler:
            scheduler.step()
        loss_history.append(epoch_loss)
        # if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save loss history to a file
    if model_name:
        file_name = f"{model_name}_training_loss.txt"
        with open(file_name, "w") as f:
            for epoch, loss in enumerate(loss_history, start=1):
                f.write(f"Epoch {epoch}: {loss:.4f}\n")

    training_time = time.time() - start_time
    return model, loss_history, training_time
