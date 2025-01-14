"""
Description:
Small CNN model for image classification
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, gray_scale=False, num_classes=10):
        super().__init__()
        # Convolutional Layer 1
        if gray_scale:
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)  # 10 output classes for CIFAR-10


    def forward(self, x):
        # Pass through conv1 and pool
        x = self.pool(torch.relu(self.conv1(x)))
        # Pass through conv2 and pool
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        # Pass through fc1 and fc2
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
  

    

