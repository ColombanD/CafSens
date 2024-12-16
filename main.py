from models.CNN import CNN
from models.Transformers import Transformer
from models.Classifier import Classifier
from catastrophic_testing import CataForgetter
from ClassiLoader import ClassiLoader
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
    liste = [[0,1,2,3,4],[5,6,7,8,9]]
    D = ClassiLoader(trainset).split(liste)
    trainloader_old, trainloader_new = D[0], D[1]

    """trainset_old, trainset_new = torch.utils.data.random_split(trainset, [40000, 10000])
    trainloader_old = DataLoader(trainset_old, batch_size=4, shuffle=True, num_workers=2)
    trainloader_new = DataLoader(trainset_new, batch_size=4, shuffle=True, num_workers=2)"""

    # Define a model
    model = CNN()
    # model = Transformer(input_dim=28, d_model=256, n_layers=6, heads=8, n_mlp=4, n_classes=10)

    # Choose between a classifier and a Regressor
    classifier = Classifier(model)

    # Initiate the CataForgetter
    cata = CataForgetter(classifier, trainloader_old, trainloader_new, True)
    CF = cata.get_CF()
    for cf in CF:
        print(cf)

if "__main__" == __name__:
    main()