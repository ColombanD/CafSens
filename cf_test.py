import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

class CFTest:
    def __init__(self, model, old_testloader, new_trainloader, new_testloader, device=None, 
                 learning_rate=0.01, momentum=0.9, criterion=None, epochs=10):
        """
        Initializes the CFTest class.

        Args:
            model (nn.Module): The neural network model.
            old_testloader (DataLoader): DataLoader for the old test dataset (CIFAR-10).
            new_trainloader (DataLoader): DataLoader for the new training dataset (5 classes from CIFAR-100).
            new_testloader (DataLoader): DataLoader for the new test dataset (5 classes from CIFAR-100).
            device (torch.device, optional): Device to run computations on. Defaults to CUDA if available.
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.01.
            momentum (float, optional): Momentum for optimizer. Defaults to 0.9.
            criterion (nn.Module, optional): Loss function. Defaults to CrossEntropyLoss.
            epochs (int, optional): Number of training epochs on the new dataset. Defaults to 10.
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.old_testloader = old_testloader
        self.new_trainloader = new_trainloader
        self.new_testloader = new_testloader
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        self.epochs = epochs

    def compute_logits(self, dataloader, output_indices=None):
        """
        Computes logits for the given DataLoader.

        Args:
            dataloader (DataLoader): The DataLoader to compute logits for.
            output_indices (list or None): Indices of output classes to consider. If None, uses all.

        Returns:
            torch.Tensor: Concatenated logits for the entire dataset.
        """
        self.model.eval()
        logits = []
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc="Computing logits"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                if output_indices is not None:
                    outputs = outputs[:, output_indices]
                logits.append(outputs.cpu())
        logits = torch.cat(logits, dim=0)
        return logits

    def train_new(self):
        """
        Trains the model on the new training dataset for the specified number of epochs.
        """
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            print(f'\nEpoch: {epoch}/{self.epochs}')
            train_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.new_trainloader, desc="Training")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Shift targets to correspond to new class indices (10-14)
                shifted_targets = targets + 10  # Assuming CIFAR-10 classes are 0-9, new classes 10-14
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, shifted_targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += shifted_targets.size(0)
                correct += predicted.eq(shifted_targets).sum().item()

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(self.new_trainloader):
                    print(f'Batch {batch_idx+1}/{len(self.new_trainloader)} | '
                          f'Loss: {train_loss/(batch_idx+1):.3f} | '
                          f'Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    def get_true_label_probs(self, probs, dataloader, output_indices=None):
        """
        Extracts the probabilities corresponding to the true labels for each sample.

        Args:
            probs (torch.Tensor): Probabilities tensor of shape [num_samples, num_classes].
            dataloader (DataLoader): DataLoader to iterate over samples and retrieve true labels.
            output_indices (list or None): Indices of output classes to consider. If None, uses all.

        Returns:
            torch.Tensor: Tensor of true label probabilities.
        """
        true_label_probs = []
        current_index = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                batch_size = targets.size(0)
                batch_probs = probs[current_index:current_index + batch_size]
                targets = targets.to('cpu')  # Ensure targets are on CPU
                if output_indices is not None:
                    # For old_testloader (CIFAR-10), output_indices are 0-9
                    # Targets are 0-9, corresponding to output_indices
                    true_probs = batch_probs[range(batch_size), targets]
                else:
                    # For new_testloader, output_indices are 10-14
                    # Targets are 0-4, shifted to 10-14
                    shifted_targets = targets + 10
                    # Since compute_logits was called with output_indices=list(range(10,15)) for new_testloader,
                    # batch_probs includes only classes 10-14 (indices 0-4)
                    true_probs = batch_probs[range(batch_size), shifted_targets - 10]
                true_label_probs.append(true_probs.cpu())
                current_index += batch_size
        true_label_probs = torch.cat(true_label_probs, dim=0)
        return true_label_probs

    def compute_forgetting_metric(self):
        """
        Computes the forgetting metric after training on the new dataset.

        Returns:
            torch.Tensor: Tensor containing the forgetting metrics for each sample.
        """
        # Step 1: Compute old_logits (only first 10 outputs for CIFAR-10)
        print("Step 1: Computing old logits...")
        old_logits = self.compute_logits(self.old_testloader, output_indices=list(range(10)))

        # Apply softmax to get probabilities
        print("Applying softmax to old logits to get probabilities...")
        old_probs = torch.softmax(old_logits, dim=1)

        # Step 2: Train the model on new dataset
        print("Step 2: Training the model on the new dataset...")
        self.train_new()

        # Step 3: Compute new_logits (only first 10 outputs for CIFAR-10)
        print("Step 3: Computing new logits on the old test dataset...")
        new_logits = self.compute_logits(self.old_testloader, output_indices=list(range(10)))

        # Apply softmax to get probabilities
        print("Applying softmax to new logits to get probabilities...")
        new_probs = torch.softmax(new_logits, dim=1)

        # Step 4: Get true label probabilities
        print("Step 4: Extracting true label probabilities from old probabilities...")
        old_true_probs = self.get_true_label_probs(old_probs, self.old_testloader, output_indices=list(range(10)))
        print("Step 4: Extracting true label probabilities from new probabilities...")
        new_true_probs = self.get_true_label_probs(new_probs, self.old_testloader, output_indices=list(range(10)))

        # Step 5: Compute forgetting metric
        print("Step 5: Computing forgetting metrics...")
        epsilon = 1e-8  # To prevent division by zero
        forgetting_metric = new_true_probs / (old_true_probs + epsilon)

        return forgetting_metric

    def run(self):
        """
        Executes the computation of the forgetting metric.

        Returns:
            torch.Tensor: Tensor containing the forgetting metrics.
        """
        forgetting_metric = self.compute_forgetting_metric()
        return forgetting_metric
