import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from caf import Caf
from Sensi import Sensitivity

### Example Model Definition ###
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.hidden = nn.Linear(28*28, 128)
        self.output = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden(x))
        return self.output(x)

def main():
    # Hyperparameters & configuration
    batch_size = 64
    epochs_old = 5
    epochs_new = 5
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Example using MNIST as old data and Fashion-MNIST as new data (just as a demonstration)
    # In practice, pick your own datasets.
    transform = transforms.Compose([transforms.ToTensor()])
    
    old_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    new_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

    # Split old_dataset and new_dataset into train/test parts
    old_size = len(old_dataset)
    old_train_size = old_size // 2
    old_test_size = old_size - old_train_size
    train_old_dataset, test_old_dataset = random_split(old_dataset, [old_train_size, old_test_size])

    new_size = len(new_dataset)
    new_train_size = new_size // 2
    new_test_size = new_size - new_train_size
    train_new_dataset, test_new_dataset = random_split(new_dataset, [new_train_size, new_test_size])

    train_old_loader = DataLoader(train_old_dataset, batch_size=batch_size, shuffle=True)
    test_old_loader = DataLoader(test_old_dataset, batch_size=batch_size, shuffle=False)

    train_new_loader = DataLoader(train_new_dataset, batch_size=batch_size, shuffle=True)
    test_new_loader = DataLoader(test_new_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    num_classes = 10  # both MNIST and FashionMNIST have 10 classes
    model = SimpleModel(num_classes=num_classes).to(device)

    # Initialize Caf experiment
    caf = Caf(model, train_old_loader, test_old_loader, train_new_loader, test_new_loader, device=device)

    # 1. Train on old
    print("Training on old dataset...")
    caf.train_old(epochs=epochs_old, lr=lr)

    # 2. Test on old
    acc_old_before = caf.test_old()
    caf.history["test_old_acc_before"] = acc_old_before

    # 3. Get old_true_logits
    old_true_logits = caf.get_true_probs()

    # 4. Train on new
    print("Training on new dataset...")
    caf.train_new(epochs=epochs_new, lr=lr)

    # 5. Test on new
    acc_new = caf.test_new()
    caf.history["test_new_acc"] = acc_new

    # 6. Test on old again to see if forgetting occurred
    acc_old_after = caf.test_old()
    caf.history["test_old_acc_after"] = acc_old_after

    # 7. Get new_true_logits
    new_true_logits = caf.get_true_probs()

    # Compute ratio
    # Make sure no division by zero occurs. If old_true_logits are zero, consider a small epsilon.
    epsilon = 1e-10
    ratio = new_true_logits / (old_true_logits + epsilon)

    # ratio now is the element-wise ratio [test_old, new_true_logit/old_true_logit]
    # This can be saved or analyzed further
    print("Ratio of new_true_logit/old_true_logit for test_old samples:", ratio)

    # Print a summary
    print("Summary:")
    print(f"Old accuracy before training new: {acc_old_before:.4f}")
    print(f"New accuracy after training new: {acc_new:.4f}")
    print(f"Old accuracy after training new: {acc_old_after:.4f}")

    # Analyze catastrophic forgetting: If old accuracy drops significantly and ratio < 1,
    # that suggests catastrophic forgetting.



    global trainloader
    trainloader = train_old_loader  # needed for Sensitivity class to infer num_classes

    # Compute sensitivities for each sample in test_old
    sens = Sensitivity(model, test_old_loader, device=device)
    sensitivities = sens.get_sensitivities()  # D_sens = [test_old, sensitivities]

    # Check dimensions
    print("D_cf shape:", ratio.shape)
    print("D_sens shape:", sensitivities.shape)

    # Convert ratio and sensitivities to NumPy for analysis
    D_cf = ratio.cpu().numpy() if torch.is_tensor(ratio) else ratio
    D_sens = np.array(sensitivities)

    # 4. Output a tensor/array of the form [test_old, sensitivities]
    # D_sens corresponds to test_old samples in order. If you need a tensor:
    D_sens_tensor = torch.tensor(D_sens)

    # Analyze relation between D_cf and D_sens
    # 1. Plot D_cf vs D_sens
    plt.figure(figsize=(8,6))
    plt.scatter(D_cf, D_sens, alpha=0.5)
    plt.xlabel('Catastrophic Forgetting (D_cf)')
    plt.ylabel('Sensitivity (D_sens)')
    plt.title('D_cf vs D_sens')
    plt.grid(True)
    plt.show()

    # 2. Compute some summary statistics (correlation)
    corr, p_value = pearsonr(D_cf, D_sens)
    print(f"Pearson correlation between D_cf and D_sens: {corr:.4f} (p-value: {p_value:.4e})")

    print("D_cf mean:", np.mean(D_cf), "D_cf median:", np.median(D_cf))
    print("D_sens mean:", np.mean(D_sens), "D_sens median:", np.median(D_sens))

    # 3. Draw conclusions based on the correlation and the plot
    if corr > 0.3:
        print("Moderate positive correlation: samples with higher sensitivity tend to be forgotten more.")
    elif corr < -0.3:
        print("Moderate negative correlation: samples with higher sensitivity tend to be forgotten less.")
    else:
        print("Weak correlation: no strong linear relationship between sensitivity and catastrophic forgetting.")

if __name__ == "__main__":
    main()
