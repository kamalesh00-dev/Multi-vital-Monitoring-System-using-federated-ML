import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ======================================================
# 1) GLOBAL SETTINGS
# ======================================================
DATA_PATH = os.path.join("data", "real", "Health_data.csv")


SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
EPOCHS = 5
ROUNDS = 30
CLIENTS = 3
LR = 1e-3
BATCH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)

# ======================================================
# 2) LOAD DATASET (pulse, body_temperature, SpO2, Status)
# ======================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Using dataset: {DATA_PATH}")

# Normalize column names
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Expect a target column named "Status"
if "Status" not in df.columns:
    raise ValueError(f"Target column 'Status' not found. Columns: {df.columns.tolist()}")

# X = features, y = target
X = df.drop(columns=["Status"])
y_raw = df["Status"].values

# If Status is not numeric 0/1, encode it
if df["Status"].dtype.kind not in "iu":  # not integer/unsigned
    y = LabelEncoder().fit_transform(y_raw)
else:
    # ensure 0/1 ints
    y = df["Status"].astype(int).values

# Train/Test split BEFORE scaling to avoid leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=SEED, stratify=y
    if len(np.unique(y)) > 1 else None
)

# Scale using TRAIN ONLY
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Split train among clients (index splits)
idx_splits = np.array_split(np.arange(len(X_train)), CLIENTS)
client_data = [(X_train[idx], y_train[idx]) for idx in idx_splits]
client_sizes = [len(s[0]) for s in client_data]

print("Client sizes:", client_sizes)

# ======================================================
# 3) MODEL
# ======================================================
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]
num_classes = int(np.max(y) + 1)  # assumes labels 0..K-1
if num_classes < 2:
    raise ValueError("Target must have at least 2 classes for classification.")
global_model = SimpleNN(input_dim, num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()

# ======================================================
# 4) HELPERS
# ======================================================
def to_tensor(x, y):
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return xt, yt

def client_train(model, X, y, epochs=EPOCHS, lr=LR, batch=BATCH):
    """Mini-batch local training and return state_dict + last loss."""
    model = model.to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    xt, yt = to_tensor(X, y)
    dataset = torch.utils.data.TensorDataset(xt, yt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

    last_loss = 0.0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    # return CPU state_dict for safe averaging
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, last_loss

def evaluate(model, X, y):
    model = model.to(DEVICE)
    model.eval()
    xt, _ = to_tensor(X, y)
    with torch.no_grad():
        logits = model(xt.to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, labels=np.arange(num_classes))
    return acc, cm

def set_state(model, state):
    model.load_state_dict(state)

def fedavg(states, weights):
    """Weighted FedAvg over list of state_dicts using weights (e.g., client sizes)."""
    if len(states) == 1:
        return states[0]
    total_w = float(np.sum(weights))
    avg_state = {}
    keys = states[0].keys()
    for k in keys:
        s = None
        for st, w in zip(states, weights):
            contrib = st[k] * (w / total_w)
            s = contrib if s is None else s + contrib
        avg_state[k] = s
    return avg_state

# ======================================================
# 5) TRAIN LOOP (FEDERATED)
# ======================================================
training_history = {"rounds": [], "accuracy": [], "loss": [], "clients": {}}
for i in range(CLIENTS):
    training_history["clients"][f"Client_{i+1}"] = {"accuracy": []}

client_metrics_records = []  # for client_metrics.csv

print("\nStarting Federated Training...\n")

for rnd in range(1, ROUNDS + 1):
    local_states, local_losses = [], []

    print(f"---------------- Round {rnd:02d} ----------------")
    for i, (Xc, yc) in enumerate(client_data, start=1):
        # start from current global
        local = SimpleNN(input_dim, num_classes)
        set_state(local, global_model.state_dict())

        state_i, loss_i = client_train(local, Xc, yc)
        acc_i, _ = evaluate(local, Xc, yc)

        local_states.append(state_i)
        local_losses.append(loss_i)
        training_history["clients"][f"Client_{i}"]["accuracy"].append(float(acc_i))

        client_metrics_records.append({
            "Round": rnd,
            "Client": f"Client_{i}",
            "TrainSamples": int(len(Xc)),
            "LocalLoss": float(loss_i),
            "LocalAcc": float(acc_i)
        })

        print(f"Client {i}: samples={len(Xc):4d} | Acc={acc_i:.4f} | Loss={loss_i:.4f}")

    # FedAvg with sample-size weights
    new_state = fedavg(local_states, client_sizes)
    set_state(global_model, new_state)

    # Evaluate on test
    g_acc, g_cm = evaluate(global_model, X_test, y_test)
    avg_loss = float(np.mean(local_losses))
    print(f"Global Acc: {g_acc:.4f} | Avg Local Loss: {avg_loss:.4f}")

    # Log & save progress
    training_history["rounds"].append(rnd)
    training_history["accuracy"].append(float(g_acc))
    training_history["loss"].append(avg_loss)

    # Save frequently so the dashboard can pick it up while training
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)
    np.save(os.path.join(SAVE_DIR, "confusion_matrix.npy"), g_cm)

# ======================================================
# 6) SAVE FINAL ARTIFACTS
# ======================================================
torch.save(global_model.state_dict(), os.path.join(SAVE_DIR, "global_simpleNN.pt"))

# per-client metrics CSV (for dashboard plots)
pd.DataFrame(client_metrics_records).to_csv(
    os.path.join(SAVE_DIR, "client_metrics.csv"), index=False
)

print("\nTraining complete ✅")
print(f"Final Global Accuracy: {training_history['accuracy'][-1]:.4f}")
print("Results saved in 'saved_models/' directory.")
