import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkImageCNN(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(NetworkImageCNN, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3, out_channels=16 , kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(in_channels=16, out_channels=32 , kernel_size=3, stride=1, padding=1)
        self.conv3=nn.Conv2d(in_channels=32, out_channels=64 , kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1 , 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class Conv1DCNN(nn.Module):
    def __init__(self):
        super(Conv1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)  
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Adjust the input size for batch normalization
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)  # Dropout layer
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2, 64)  # Adjusting for new dimensions
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.dropout(x)  # Apply dropout after convolutions
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output layer
        return x

class SimpleFNN(nn.Module):
    def __init__(self):
        super(SimpleFNN, self).__init__()
        
        self.fc1 = nn.Linear(14, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x
    
def train(model, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.BCEWithLogitsLoss() 
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for input, labels in trainloader:
            input, labels = input.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(input).squeeze(1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        epoch_loss = loss / len(trainloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


def test(model, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.BCEWithLogitsLoss() 
    correct, loss = 0, 0.0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            input, labels = data[0].to(device), data[1].to(device)

            output = model(input).squeeze(1)
            loss += criterion(output, labels).item()
            predictions = (output > 0.5).int()  # Apply threshold for binary prediction
            correct += (predictions == labels.int()).sum().item()

    average_loss = loss / len(testloader)
    accuracy = correct / len(testloader.dataset)
    return average_loss, accuracy