import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 output classes for CIFAR-10
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        # Pass through conv1 and pool
        x = self.pool(torch.relu(self.conv1(x)))
        # Pass through conv2 and pool
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        # Pass through fc1 and fc2
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train(self, dataloader):
        # Training the model
        num_epochs = 1
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 1000 == 999:    # Print every 1000 mini-batches, where here batch = mini-batch
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}")
                    running_loss = 0.0

        print('Finished Training')

    def get_probability_for_true_class(self, x, y):
        prob_list = torch.softmax(self.forward(x), dim=1)
        return prob_list[y]

