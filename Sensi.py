# Estimer la sensitivity d'un model aux inputs de l'entiereté d'un dataset
    # Train le model sur tout le dataset
    # Get les probs de la true class
    # Get la variance du posterior par variational inference
    # Utiliser la formule: residual * lambdas * variance Eq.12

import numpy as np
from laplace import Laplace
from laplace.curvature import AsdlGGN
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


class Sensitivity:
    def __init__(self, model, dataloader, device='cuda'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
   
    def compute_variance_infer(self):
        """
        Variational Inference to estimate posterior variance.
        """
        print("In laplace")
        laplace_object = Laplace(
            self.model, 'classification',
            subset_of_weights='last_layer',
            hessian_structure='diag',
            backend=AsdlGGN)
        print("In laplace created")

        laplace_object.fit(self.dataloader)
        print("In after laplace fit")

        fvars = []
        for inputs, _ in tqdm(self.dataloader, desc="Computing Variance"):
            inputs = inputs.to(self.device)
            _, fvar = laplace_object._glm_predictive_distribution(inputs)
            fvars.append(np.diagonal(fvar.cpu().numpy(), axis1=1, axis2=2))

        del laplace_object
        torch.cuda.empty_cache()

        return np.vstack(fvars)


    def softmax_hessian(self, probs, eps=1e-10):
        """
        Approximation of softmax Hessian diagonal, it is a diagonal matrix: diag(p * (1-p)).
        """
        return torch.clamp(probs * (1 - probs), min=eps)


    def get_sensitivities(self):
        self.model.eval()
        # Calculate num_classes by iterating through the dataloader
        all_labels = []
        for _, y in trainloader:
            all_labels.extend(y.cpu().tolist())

        num_classes = max(all_labels) + 1

        residuals_list = []
        lambdas_list = []

        for X, y in tqdm(self.dataloader, desc="Computing Residuals and Lambdas"):
            X, y = X.to(self.device), y.to(self.device)

            # Get model predictions
            logits = self.model(X)
            probs = F.softmax(logits, dim=-1)

            # Compute residuals
            one_hot_y = F.one_hot(y, num_classes).float()  # Ensure data type compatibility
            residuals_list.append((probs - one_hot_y).cpu().detach().numpy())

            # Compute the Hessian approximation
            lambdas = self.softmax_hessian(probs).cpu().detach().numpy()
            lambdas_list.append(lambdas)

        residuals = np.vstack(residuals_list)
        lambdas = np.vstack(lambdas_list)

        # Use computed residuals and lambdas
        print("Yes")
        vars = self.compute_variance_infer()
        print("No")

        # Compute sensitivities
        sensitivities = lambdas * vars * residuals
        sensitivities = np.sum(np.abs(sensitivities), axis=-1)

        return sensitivities


# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

# Instantiate model, loss, and optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
model = SimpleNet(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(1):  # Train for 2 epochs
    model.train()
    running_loss = 0.0

    epoch_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}", unit="batch")

    for i, (inputs, labels) in enumerate(epoch_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update the description with the running loss
        epoch_bar.set_postfix(loss=running_loss / (i + 1))

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

sensitivity_analyzer = Sensitivity(model, trainloader, device=device)

# Compute sensitivities
sensitivities = sensitivity_analyzer.get_sensitivities()

print("Sensitivities:", sensitivities)
print("sensitivities shape:", sensitivities.shape)