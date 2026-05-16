import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================================
# CONFIGURATION
# ==========================================================
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 5
ROUNDS = 20
LR = 1e-3
BATCH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# MODEL
# ==========================================================
class MLP(nn.Module):
    def __init__(self, d_in, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def prep_dataset(path, target_col="Status"):
    """Load and preprocess dataset for federated ML"""
    try:
        df = pd.read_csv(path, sep=",", on_bad_lines="skip", engine="python")
    except Exception as e:
        print(f"⚠️ Error reading {path}: {e}")
        return np.array([]), np.array([])

    # Clean column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Extract target
    if target_col in df.columns:
        y = df[target_col]
        if not np.issubdtype(y.dtype, np.number):
            y = LabelEncoder().fit_transform(y)
    else:
        y = df.iloc[:, -1].values

    X = df.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number])
    X = X.values.astype(np.float32)
    y = np.array(y, dtype=np.int64)

    # --- Map arrhythmia classes to risk levels ---
    if "arrhythmia" in path.lower():
        # Simplify multi-class disease labels to 3 risk categories
        y_risk = np.zeros_like(y)
        y_risk[y <= 3] = 0        # Normal / mild
        y_risk[(y > 3) & (y <= 7)] = 1  # Moderate
        y_risk[y > 7] = 2         # Severe / critical
        y = y_risk

    return X, y


def pad_to_max_dim(X, D):
    """Pads or trims dataset to consistent feature dimension D"""
    if X.shape[1] == D:
        return X
    elif X.shape[1] < D:
        Z = np.zeros((X.shape[0], D), dtype=np.float32)
        Z[:, :X.shape[1]] = X
        return Z
    else:
        return X[:, :D]


def train_local(model, X, y, epochs=EPOCHS, lr=LR):
    """Trains local client model"""
    model = model.to(DEVICE)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    data = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.tensor(X).to(DEVICE)), dim=1).cpu().numpy()
    acc = accuracy_score(y, preds)
    return model.state_dict(), acc, float(loss.item())


def fedavg(states, weights):
    """Federated averaging algorithm"""
    total = float(sum(weights))
    avg = {}
    for k in states[0].keys():
        avg[k] = sum([s[k] * (w / total) for s, w in zip(states, weights)])
    return avg


# ==========================================================
# LOAD DATASETS
# ==========================================================
datasets = {}
paths = {
    "HeartVitals": "data/real/Health_data.csv",
    "MITBIH": "datasets/health_status/kaggle_heartbeat/mitbih_test.csv",
    "Arrhythmia": "datasets/health_status/arrhythmia/arrhythmia_clean.csv"
}

for name, p in paths.items():
    if os.path.exists(p):
        X, y = prep_dataset(p)
        if X.size > 0:
            datasets[name] = (X, y)
    else:
        print(f"⚠️ {name} dataset not found — skipping.")

# ==========================================================
# TRAINING + HELD-OUT TEST SETUP
# ==========================================================
train_clients = {k: v for k, v in datasets.items() if k != "Arrhythmia"}
heldout_name = "Arrhythmia" if "Arrhythmia" in datasets else None
heldout_data = datasets.get(heldout_name, (None, None))

print("\n🔍 Training clients:")
for k, (X, y) in train_clients.items():
    print(f" - {k:10s} | n={len(X):6d} | features={X.shape[1]} | class dist={np.unique(y, return_counts=True)[1].tolist()}")

if heldout_name:
    Xh, yh = heldout_data
    print(f"🧪 Held-out test: {heldout_name} | n={len(Xh):6d} | features={Xh.shape[1]} | class dist={np.unique(yh, return_counts=True)[1].tolist()}")

# ==========================================================
# UNIFY FEATURE DIMENSIONS
# ==========================================================
D_MAX = max([X.shape[1] for X, _ in datasets.values()])
print(f"\n📐 Unified input dimension across all datasets: {D_MAX}")

for k in datasets.keys():
    X, y = datasets[k]
    datasets[k] = (pad_to_max_dim(X, D_MAX), y)

train_clients = {k: v for k, v in datasets.items() if k != "Arrhythmia"}
X_held, y_held = datasets["Arrhythmia"] if "Arrhythmia" in datasets else (None, None)

# ==========================================================
# FEDERATED TRAINING
# ==========================================================
global_model = MLP(D_MAX, 3).to(DEVICE)
print("\n🚀 Cross-domain Federated Training (train on HeartVitals + MITBIH)")

hist = {"rounds": [], "acc": [], "loss": []}

for rnd in range(1, ROUNDS + 1):
    local_states, local_acc, local_loss, weights = [], [], [], []

    print(f"----- Round {rnd:02d} -----")
    for name, (X, y) in train_clients.items():
        model = MLP(D_MAX, 3)
        model.load_state_dict(global_model.state_dict())
        state, acc, loss = train_local(model, X, y)
        local_states.append(state)
        local_acc.append(acc)
        local_loss.append(loss)
        weights.append(len(X))
        print(f"  {name:10s} | n={len(X):6d} | TrainAcc={acc:.4f} | Loss={loss:.4f}")

    global_state = fedavg(local_states, weights)
    global_model.load_state_dict(global_state)

    hist["rounds"].append(rnd)
    hist["acc"].append(np.mean(local_acc))
    hist["loss"].append(np.mean(local_loss))
    print(f"  ➜ Proxy Global Acc (train domain): {np.mean(local_acc):.4f} | Avg Local Loss: {np.mean(local_loss):.4f}")

# ==========================================================
# HELD-OUT CROSS-DOMAIN TEST
# ==========================================================
if X_held is not None:
    print("\n================ HELD-OUT CROSS-DOMAIN TEST ================")
    global_model.eval()
    with torch.no_grad():
        preds = torch.argmax(global_model(torch.tensor(X_held).to(DEVICE)), dim=1).cpu().numpy()

    acc_ho = accuracy_score(y_held, preds)
    cm_ho = confusion_matrix(y_held, preds)
    report = classification_report(y_held, preds, output_dict=True)

    print(f"🧪 Held-out Test on {heldout_name} Acc: {acc_ho:.4f}")
    print("Confusion Matrix:\n", cm_ho)
    print("Classification Report:\n", classification_report(y_held, preds))

    np.save(os.path.join(SAVE_DIR, "confusion_matrix_crossdomain.npy"), cm_ho)
    with open(os.path.join(SAVE_DIR, "classification_report_crossdomain.json"), "w") as f:
        json.dump(report, f, indent=2)

# ==========================================================
# SAVE GLOBAL MODEL + HISTORY
# ==========================================================
torch.save(global_model.state_dict(), os.path.join(SAVE_DIR, "global_health_risk_model_crossdomain.pt"))
with open(os.path.join(SAVE_DIR, "training_history_crossdomain.json"), "w") as f:
    json.dump(hist, f, indent=2)

print(f"\n✅ Cross-domain training complete.")
print(f"Results saved in '{SAVE_DIR}' folder.")

# ==========================================================
# 🔧 FINE-TUNE GLOBAL MODEL ON COMBINED DATA
# ==========================================================
print("\n🔧 Fine-tuning global model on combined dataset (transfer learning)...")

all_X = np.vstack([datasets[k][0] for k in datasets.keys()])
all_y = np.concatenate([datasets[k][1] for k in datasets.keys()])

global_model = MLP(D_MAX, 3).to(DEVICE)
global_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "global_health_risk_model_crossdomain.pt")))

state, acc, loss = train_local(global_model, all_X, all_y, epochs=10)
torch.save(global_model.state_dict(), os.path.join(SAVE_DIR, "global_health_risk_model_finetuned.pt"))

print(f"✅ Fine-tuned model saved — Combined Acc={acc:.4f}, Loss={loss:.4f}")
