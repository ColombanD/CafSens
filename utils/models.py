"""
Description: 
This file contains the function to load the model based on the model name.
"""

from torchvision import models
from models.CNN import CNN
from models.Transformer import Transformer
import torch.nn as nn

def load_model(model_name, gray_scale, num_classes):
    """Returns the model based on the model name."""
    
    if model_name == 'CNN':
        return CNN(gray_scale=gray_scale, num_classes=num_classes)
    elif model_name == 'Transformer':
        return Transformer(grayscale=gray_scale)
    elif model_name == 'AlexNet':
        return models.alexnet(pretrained=False, num_classes=num_classes)
    elif model_name == 'Resnet18':
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        if gray_scale:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model