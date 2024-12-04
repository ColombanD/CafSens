import models
import catastrophic_testing
import torch
import torchvision
import torchvision.transforms as transforms

# Dummy test
def main():
    # Load and normalize CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # split the training set into two parts, old and new
    trainset_old, trainset_new = torch.utils.data.random_split(trainset, [40000, 10000])

    model = models.CNN()
    cata = catastrophic_testing.CataForgetter(model, trainset_old, trainset_new)
    CF = cata.get_CF()
    print(CF)
