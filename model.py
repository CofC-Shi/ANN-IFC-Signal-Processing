import torch
import torch.nn as nn

class CustomModelTSF(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        """
        Initialize the CustomModel for time series forecasting.

        Args:
            input_dim (int): Number of input features (sliding window size).
            output_dim (int): Number of output features (forecasting steps). Default is 1 (one-step-ahead forecasting).
        """
        super(CustomModelTSF, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_dim).
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x