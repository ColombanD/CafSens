import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from cf_test import CFTest  # Importing the CFTest class from cf_test.py

def progress_bar(current, total, msg=None):
    if msg:
        print(f"{current}/{total} - {msg}")

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Define transformations for CIFAR-10 and CIFAR-100
    transform_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load CIFAR-10 test dataset as old_testloader
    old_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_cifar10)
    old_testloader = DataLoader(old_testset, batch_size=100, shuffle=False, num_workers=2)

    # Create new_trainset and new_testset with 5 selected classes from CIFAR-100
    # Define selected classes
    selected_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver']

    # Load CIFAR-100 datasets
    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cifar100)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_cifar100)

    # Get class indices for selected classes
    class_to_idx = cifar100_train.class_to_idx
    selected_class_indices = [class_to_idx[class_name] for class_name in selected_classes]

    # Create a mapping from original labels to new labels (0-4)
    label_mapping = {original_label: new_label for new_label, original_label in enumerate(selected_class_indices)}

    # Function to filter dataset
    def filter_dataset(dataset, selected_class_indices, label_mapping):
        indices = [i for i, label in enumerate(dataset.targets) if label in selected_class_indices]
        subset = Subset(dataset, indices)
        
        # Override __getitem__ to relabel targets
        original_getitem = subset.dataset.__getitem__
        
        def new_getitem(idx):
            img, target = original_getitem(subset.indices[idx])
            new_target = label_mapping[target]
            return img, new_target
        
        subset.__getitem__ = new_getitem
        return subset

    # Create filtered datasets
    new_trainset = filter_dataset(cifar100_train, selected_class_indices, label_mapping)
    new_testset = filter_dataset(cifar100_test, selected_class_indices, label_mapping)

    # Create DataLoaders
    new_trainloader = DataLoader(new_trainset, batch_size=100, shuffle=True, num_workers=2)
    new_testloader = DataLoader(new_testset, batch_size=100, shuffle=False, num_workers=2)

    # Initialize the model (e.g., a simple CNN)
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=15):  # Changed from 10 to 15
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 8 * 8, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    # Initialize the model with 15 classes (10 CIFAR-10 + 5 new classes)
    model = SimpleCNN(num_classes=15)
    model = model.to(device)

    # Pre-training on CIFAR-10
    # For demonstration, we'll train only the first 10 outputs corresponding to CIFAR-10
    # The last 5 outputs (10-14) correspond to the new CIFAR-100 classes

    # Initialize CIFAR-10 training DataLoader
    trainloader_cifar10 = DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10),
        batch_size=100, shuffle=True, num_workers=2)

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("Pre-training the model on CIFAR-10...")
    model.train()
    for epoch in range(5):  # Pre-train for 5 epochs
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in tqdm(trainloader_cifar10, desc=f"Pre-training Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # Compute loss only on the first 10 outputs (CIFAR-10 classes)
            loss = criterion(outputs[:, :10], targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs[:, :10].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'Epoch {epoch+1} | Loss: {running_loss/len(trainloader_cifar10):.3f} | '
              f'Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    # Initialize CFTest with the model and datasets
    cf_tester = CFTest(
        model=model,
        old_testloader=old_testloader,
        new_trainloader=new_trainloader,
        new_testloader=new_testloader,
        device=device,
        criterion=criterion,
        epochs=10  # Number of epochs to train on the new dataset
    )

    # Run the forgetting metric computation
    print("Starting forgetting metric computation...")
    forgetting_metrics = cf_tester.run()
    print("Forgetting metrics computed.")

    # Example: Print first 10 forgetting metrics
    print("First 10 forgetting metrics:", forgetting_metrics[:10])

if __name__ == "__main__":
    main()
