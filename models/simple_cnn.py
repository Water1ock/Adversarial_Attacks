# simple_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # CIFAR-10 images are 32x32, so after 3 conv layers with pooling, size is reduced to 4x4
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First layer (conv -> batch norm -> relu -> pool)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Second layer (conv -> batch norm -> relu -> pool)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Third layer (conv -> batch norm -> relu -> pool)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the tensor
        x = x.view(-1, 128 * 4 * 4)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# To create a model instance:
# model = SimpleCNN(num_classes=10)