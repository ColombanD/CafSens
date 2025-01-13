"""
Description: 
This file contains the implementation of the Caf class, which encapsulates the catastrophic forgetting experiment setup.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Caf:
    """
    Caf class encapsulating the catastrophic forgetting experiment setup.

    Attributes:
        model (nn.Module): The neural network model.
        list_train_loaders (list of DataLoaders): DataLoaders for the different training phases
        list_test_loaders (list of DataLoaders): DataLoaders for the different test phases
        device (str): The device to run computations on.
    """
    def __init__(self, model, list_train_loaders, list_test_loaders, logger, device=None):
        self.model = model
        self.list_train_loaders = list_train_loaders
        self.list_test_loaders = list_test_loaders
        self.logger = logger
        
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # For logging or debugging
        self.history = {}

    def train(self, epochs=10, lr=1e-3, train_nbr=0):
        """
        Train the model on a training set.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            loader = self.list_train_loaders[train_nbr]

            # Iterate over the dataset
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Step the scheduler: reduce the LR by a constant factor gamma every step_size epoch. For stability
            scheduler.step()
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")


    def test(self, test_nbr):
        """
        Test the model on a test set and return accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            loader = self.list_test_loaders[test_nbr]
            
            # Iterate over the dataset
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        self.logger.info(f"[test nbr {test_nbr}] Accuracy: {acc:.4f}")
        return acc  

    def get_true_probs(self, train=True, dataset_nbr=0):
        """
        Compute the probability corresponding to the true class for train or test and a specific dataset.

        Returns:
            A tensor of shape (N,) where N is the size of the dataset,
            containing the probability for the true class for each sample.
        """
        self.model.eval()
        true_probs = []
        if train:
            loader = self.list_train_loaders[dataset_nbr]
        else:
            loader = self.list_test_loaders[dataset_nbr]
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)  # shape: [batch_size, num_classes]
            
                # Convert logits to probabilities using softmax
                probs = F.softmax(outputs, dim=1)  # shape: [batch_size, num_classes]

                # Extract the probability corresponding to the true label for each sample
                batch_size = probs.size(0)
                for i in range(batch_size):
                    true_probs.append(probs[i, targets[i]].item())
        return torch.tensor(true_probs)

    def get_caf(self, old_true_probs, new_true_probs, caf_type="difference"):
        """
        Compute the catastrophic forgetting (CAF) score.

        Args:
            old_true_probs (torch.Tensor): A tensor of shape (N,) containing the true probabilities
                for the old test set.
            new_true_probs (torch.Tensor): A tensor of shape (N,) containing the true probabilities
                for the new test set.

        Returns:
            The CAF score.
        """
        eps = 1e-10
        if caf_type == "difference":
            return (old_true_probs - new_true_probs)
        return (old_true_probs / new_true_probs + eps)
