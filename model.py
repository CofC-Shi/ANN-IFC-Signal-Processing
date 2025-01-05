import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class CustomModel(nn.Module):
    def __init__(self, input_dim):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, input_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

class VGG16FineTuning(torch.nn.Module):
    def __init__(self, num_classes=32):
        super(VGG16FineTuning, self).__init__()
        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)

        # Freeze all layers except the last convolutional block
        for param in self.vgg16.features[:24].parameters():
            param.requires_grad = False

        # Replace the classifier with a custom head
        self.vgg16.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, num_classes)  # Number of output features
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x

class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(SimpleRNNModel, self).__init__()
        self.hidden_dim = hidden_dim

        # RNN layer
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)  # RNN output: [batch_size, seq_length, hidden_dim]
        out = self.fc(out[:, :])  # Use the last hidden state
        return out