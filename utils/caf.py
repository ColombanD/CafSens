# Improved implementation of CFTest
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
        train_old_loader (DataLoader): DataLoader for the old training set.
        test_old_loader (DataLoader): DataLoader for the old test set.
        train_new_loader (DataLoader): DataLoader for the new training set.
        test_new_loader (DataLoader): DataLoader for the new test set.
        device (str): The device to run computations on.
    """
    def __init__(self, model, train_old_loader, test_old_loader, train_new_loader, test_new_loader, logger, device=None):
        self.model = model
        self.train_old_loader = train_old_loader
        self.test_old_loader = test_old_loader
        self.train_new_loader = train_new_loader
        self.test_new_loader = test_new_loader
        self.logger = logger
        
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # For logging or debugging
        self.history = {
            "test_old_acc_before": None,
            "test_old_acc_after": None,
            "test_new_acc": None
        }

    def train(self, epochs=10, lr=1e-3, train_old=True):
        """
        Train the model on the either the old or the new training set.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            if train_old:
                loader = self.train_old_loader
            else:
                loader = self.train_new_loader

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

            if train_old:
                self.logger.info(f"[train_old] Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_old_loader):.4f}")
            else:
                self.logger.info(f"[train_new] Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_new_loader):.4f}")

    def test(self, test_old=True):
        """
        Test the model on either the old test set or the new test set and return accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            if test_old:
                loader = self.test_old_loader
            else:
                loader = self.test_new_loader
            
            # Iterate over the dataset
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        if test_old:
            self.logger.info(f"[test_old] Accuracy: {acc:.4f}")
        else:
            self.logger.info(f"[test_new] Accuracy: {acc:.4f}")
        return acc
    

    def get_true_probs(self, train=True):
        """
        Compute the probability corresponding to the true class for each sample in train_old.

        Returns:
            A tensor of shape (N,) where N is the size of test_old,
            containing the probability for the true class for each sample.
        """
        self.model.eval()
        true_probs = []
        if train:
            loader = self.train_old_loader
        else:
            loader = self.train_new_loader
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
        print(f'true_probs size: {len(true_probs)}')
        return torch.tensor(true_probs)

    def get_caf(self, old_true_probs, new_true_probs):
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
        return (new_true_probs / old_true_probs + eps)
