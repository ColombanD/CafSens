"""
Description:
Script for visualizing model sensitivity on different images.
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from utils.sensitivity import Sensitivity
from models.CNN import CNN
import torch.optim as optim
import torch.nn as nn


transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

# Example DataLoader
dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# model
model = CNN(gray_scale=True, num_classes=10)

# Train model
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

criterion = nn.CrossEntropyLoss()
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0

    # Iterate over the dataset
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
            

# Calculate or get sensitivities (assuming it's a list of tensors with the same length as the dataset)
sensitivity = Sensitivity(model=model, dataloader=dataloader, device=device)
sens = sensitivity.get_sensitivities()

sens = torch.tensor(sens)

# Sort sensitivities and get indices for high and low sensitivity images
high_sens_indices = torch.argsort(sens, descending=True)[:5]  # top 5 high sensitivity
low_sens_indices = torch.argsort(sens)[:5]  # top 5 low sensitivity

# Select the corresponding images
high_sens_images = [dataloader.dataset[i][0] for i in high_sens_indices]
low_sens_images = [dataloader.dataset[i][0] for i in low_sens_indices]

# Plot high sensitivity images
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, img in enumerate(high_sens_images):
    axs[i].imshow(img.squeeze(), cmap='gray')  # Remove the channel dimension and show as grayscale
    axs[i].axis('off')
    axs[i].set_title(f"High Sensitivity {i+1}")
fig.savefig("high_sensitivity_images_fm.png", dpi=300, bbox_inches='tight')  # Save the high-sensitivity plot


# Plot low sensitivity images
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, img in enumerate(low_sens_images):
    axs[i].imshow(img.squeeze(), cmap='gray')  # Remove the channel dimension and show as grayscale
    axs[i].axis('off')
    axs[i].set_title(f"Low Sensitivity {i+1}")
fig.savefig("low_sensitivity_images_fm.png", dpi=300, bbox_inches='tight')  # Save the low-sensitivity plot


plt.show()
