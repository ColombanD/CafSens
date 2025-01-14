"""Description: 
This file contains the function to load the transforms for the models
"""

from torchvision import transforms

def load_transform(model_name):
    if model_name == 'Resnet18':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    if model_name == 'CNN':
        return transforms.Compose([
            #for compatibility between datasets and models
            transforms.Resize((28,28)),
            transforms.ToTensor()
            ])
    if model_name == 'Resnet50':
        return transforms.Compose([
            transforms.Resize(256),                # Resize the shortest side to 256 pixels
            transforms.CenterCrop(224),            # Crop the center to a 224x224 square
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])