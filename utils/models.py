"""
Description: 
This file contains the function to load the model based on the model name.
"""

from torchvision import models
from models.CNN import CNN
import torch.nn as nn

def load_model(model_name, gray_scale, num_classes):
    """Returns the model based on the model name."""
    
    if model_name == 'CNN':
        return CNN(gray_scale=gray_scale, num_classes=num_classes)
    elif model_name == 'Resnet18':
        model = models.resnet18(num_classes=num_classes)
        if gray_scale:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif model_name == 'Resnet50':
        model = models.resnet50(num_classes=num_classes)
        if gray_scale:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model