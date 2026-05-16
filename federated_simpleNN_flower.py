# federated_simpleNN_flower.py

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1️⃣ Define your SimpleNN model
# -------------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# 2️⃣ Synthetic data generation
# -------------------------------
def generate_synthetic_data(num_samples=1000, input_dim=10, num_classes=2):
    X = np.random.rand(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples).astype(np.int64)
    return X, y

# -------------------------------
# 3️⃣ Define Flower client
# -------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data, lr=0.01):
        self.model = model
        self.train_X, self.train_y = train_data
        self.val_X, self.val_y = val_data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    # Return model parameters as a list of NumPy arrays
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    # Set model parameters from a list of NumPy arrays
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    # Train the model locally
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        X = torch.tensor(self.train_X)
        y = torch.tensor(self.train_y)
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        return self.get_parameters({}), len(self.train_X), {}

    # Evaluate the model
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        X = torch.tensor(self.val_X)
        y = torch.tensor(self.val_y)
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y).item()
            preds = torch.argmax(outputs, axis=1)
            acc = (preds == y).sum().item() / len(y)
        return loss, len(y), {"accuracy": acc}

# -------------------------------
# 4️⃣ Federated training simulation
# -------------------------------
def main():
    # Number of clients
    num_clients = 3

    # Generate synthetic data for each client
    clients = []
    for i in range(num_clients):
        train_data = generate_synthetic_data()
        val_data = generate_synthetic_data(num_samples=200)
        model = SimpleNN()
        clients.append(FlowerClient(model, train_data, val_data))

    # Define a function to start simulation
    def client_fn(cid: str):
        return clients[int(cid)]

    # Flower strategy (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # All clients participate each round
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    # Start simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # -------------------------------
    # 5️⃣ Log metrics and plot
    # -------------------------------
    rounds = range(1, len(hist.metrics_centralized["accuracy"]) + 1)
    acc = [hist.metrics_centralized["accuracy"][r] for r in rounds]

    plt.plot(rounds, acc, marker='o', label='Federated Accuracy')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Learning Accuracy per Round")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------
    # 6️⃣ Save global model
    # -------------------------------
    global_model = SimpleNN()
    # Load parameters from last round
    last_params = hist.strategy_result.last_parameters
    params_dict = zip(global_model.state_dict().keys(), last_params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    global_model.load_state_dict(state_dict, strict=True)
    os.makedirs("saved_models", exist_ok=True)
    torch.save(global_model.state_dict(), "saved_models/global_simpleNN.pt")
    print("✅ Global model saved to saved_models/global_simpleNN.pt")

if __name__ == "__main__":
    main()
