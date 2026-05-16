import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNN
from data_utils import load_data

# Load data
X_train, X_test, y_train, y_test = load_data()

# Model
model = SimpleNN(input_size=10, hidden_size=16, output_size=2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Test accuracy
with torch.no_grad():
    predictions = model(X_test).argmax(dim=1)
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")

