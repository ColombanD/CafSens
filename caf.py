# Improved implementation of CFTest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
    def __init__(self, model, train_old_loader, test_old_loader, train_new_loader, test_new_loader, device=None):
        self.model = model
        self.train_old_loader = train_old_loader
        self.test_old_loader = test_old_loader
        self.train_new_loader = train_new_loader
        self.test_new_loader = test_new_loader
        
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # For logging or debugging
        self.history = {
            "test_old_acc_before": None,
            "test_old_acc_after": None,
            "test_new_acc": None
        }

    def train_old(self, epochs=10, lr=1e-3):
        """
        Train the model on the old training set.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in self.train_old_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"[train_old] Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_old_loader):.4f}")

    def test_old(self):
        """
        Test the model on the old test set and return accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.test_old_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        print(f"[test_old] Accuracy: {acc:.4f}")
        return acc

    def train_new(self, epochs=10, lr=1e-3):
        """
        Train the model on the new training set.
        """
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in self.train_new_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"[train_new] Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_new_loader):.4f}")

    def test_new(self):
        """
        Test the model on the new test set and return accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.test_new_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        acc = correct / total
        print(f"[test_new] Accuracy: {acc:.4f}")
        return acc

    def get_true_logits(self):
        """
        Compute the true logits for each sample in test_old.

        Returns:
            A tensor of shape (N,) where N is the size of test_old,
            containing the logit corresponding to the true class for each sample.
        """
        self.model.eval()
        true_logits = []
        with torch.no_grad():
            for inputs, targets in self.test_old_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)  # shape: [batch_size, num_classes]
                batch_size = outputs.size(0)
                # Extract the logit corresponding to the true label
                # outputs[i, targets[i]] is the logit for the true label of the i-th sample
                for i in range(batch_size):
                    true_logits.append(outputs[i, targets[i]].item())
        return torch.tensor(true_logits)
