from torchvision import transforms

def load_transform(model_name):
    if model_name == 'Resnet18' or model_name == 'MobileNetV2':
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