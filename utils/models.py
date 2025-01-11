from torchvision import models
from models.CNN import CNN
from models.Transformer import Transformer

def load_model(model_name, gray_scale, num_classes):
    if model_name == 'CNN':
        return CNN(gray_scale=gray_scale, num_classes=num_classes)
    elif model_name == 'Transformer':
        return Transformer(grayscale=gray_scale)
    elif model_name == 'AlexNet':
        return models.alexnet(pretrained=False, num_classes=num_classes)
    elif model_name == 'Resnet18':
        return models.rensnet18(pretrained=False, num_classes=num_classes)