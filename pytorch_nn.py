import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
data = pd.read_csv("train.csv").values
torch.manual_seed(1)

X = torch.tensor(data[1:, 1:], dtype=torch.float32) / 255.0
Y = torch.tensor(data[1:, 0], dtype=torch.long)

X_dev, Y_dev = X[:1000], Y[:1000]
X_train, Y_train = X[1000:], Y[1000:]

train_dataset = TensorDataset(X_train, Y_train)
dev_dataset = TensorDataset(X_dev, Y_dev)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.LeakyReLU()  # Changed to LeakyReLU
        self.dropout = nn.Dropout(0.3)  # Added dropout
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x  # Removed Softmax

# Initialize model, loss, and optimizer
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Changed optimizer and learning rate

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == Y_batch).sum().item()
        accuracy = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

# Validation loop
def validate_model(model, dev_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, Y_batch in dev_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            correct += (outputs.argmax(1) == Y_batch).sum().item()
    accuracy = correct / len(dev_loader.dataset)
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

# Train and validate the model
if __name__ == "__main__":
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    validate_model(model, dev_loader)