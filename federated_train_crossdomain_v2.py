import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

# Suppress scikit-learn warnings about zero-division
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# ==========================================================
# CONFIG
# ==========================================================
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 5
ROUNDS = 50  # Increased rounds for better convergence
LR = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization
BATCH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print(f"Using device: {DEVICE}")

# ==========================================================
# MODEL (Added Dropout Layers)
# ==========================================================
class MLP(nn.Module):
    def __init__(self, d_in, n_classes=3):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added Dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Added Dropout
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================================
# HELPERS
# ==========================================================
def prep_hearvitals(path, target_col="Status"):
    try:
        df = pd.read_csv(path, sep=",", on_bad_lines="skip", engine="python")
    except FileNotFoundError:
        print(f"Warning: HeartVitals file not found at {path}")
        return None, None
        
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"HeartVitals file is missing '{target_col}' column. Found: {df.columns.tolist()}")

    y = df[target_col]
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    X = df.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number])
    return X.values.astype(np.float32), y.values.astype(np.int64)

def load_mitbih(path):
    try:
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        print(f"Warning: MIT-BIH file not found at {path}")
        return None, None
        
    if df.shape[1] < 2:
        raise ValueError("MITBIH file has fewer than 2 columns; expected features + label")
    
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].astype(int).values
    
    # Map to 3 risk classes: 0,1->Normal (0); 2,3->Moderate (1); 4->Critical (2)
    mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
    y = np.array([mapping.get(int(v), 1) for v in y_raw], dtype=np.int64)
    return X, y

def load_arrhythmia(path):
    try:
        df = pd.read_csv(path, sep=",", on_bad_lines="skip", engine="python")
    except FileNotFoundError:
        print(f"Warning: Arrhythmia file not found at {path}")
        return None, None

    if 'class' in df.columns:
        y = df['class'].astype(int).values
        X = df.drop(columns=['class']).select_dtypes(include=[np.number]).values
    else:
        y = df.iloc[:, -1].astype(int).values
        X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    return X.astype(np.float32), y.astype(np.int64)

def pad_to_max_dim(X, D):
    if X.shape[1] == D:
        return X
    elif X.shape[1] < D:
        Z = np.zeros((X.shape[0], D), dtype=np.float32)
        Z[:, :X.shape[1]] = X
        return Z
    else: # Truncate if more features than D
        return X[:, :D]

def train_local(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LR):
    model = model.to(DEVICE)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    
    # --- THIS IS THE CRITICAL FIX ---
    # Convert pandas objects (if they are) to simple numpy arrays
    # This strips any lingering pandas indices that cause the KeyError.
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values.astype(np.float32)
    else:
        X_train_np = np.ascontiguousarray(X_train, dtype=np.float32)

    if isinstance(y_train, pd.Series):
        y_train_np = y_train.values.astype(np.int64)
    else:
        y_train_np = np.ascontiguousarray(y_train, dtype=np.int64)

    if isinstance(X_val, pd.DataFrame):
        X_val_np = X_val.values.astype(np.float32)
    else:
        X_val_np = np.ascontiguousarray(X_val, dtype=np.float32)
        
    if isinstance(y_val, pd.Series):
        y_val_np = y_val.values.astype(np.int64)
    else:
        y_val_np = np.ascontiguousarray(y_val, dtype=np.int64)
    
    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train_np).to(DEVICE)
    y_train_tensor = torch.tensor(y_train_np).to(DEVICE)
    X_val_tensor = torch.tensor(X_val_np).to(DEVICE)
    y_val_tensor = torch.tensor(y_val_np).to(DEVICE)
    # --- END OF FIX ---

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), # Use the new tensors
        batch_size=BATCH, shuffle=True
    )

    epoch_losses = []
    for _ in range(epochs):
        for xb, yb in loader:
            # Data is already on DEVICE
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

    model.eval()
    with torch.no_grad():
        # Also use the tensors for evaluation
        tr_preds = torch.argmax(model(X_train_tensor), dim=1).cpu().numpy()
        va_preds = torch.argmax(model(X_val_tensor), dim=1).cpu().numpy()
        
    tr_acc = accuracy_score(y_train_np, tr_preds)
    va_acc = accuracy_score(y_val_np, va_preds)
    
    return model.state_dict(), tr_acc, va_acc, avg_loss

def fedavg(states, weights):
    """Weighted average of model state dictionaries."""
    total_weight = float(sum(weights))
    if total_weight == 0:
        return states[0]  # Or handle as an error, but this shouldn't be called if no states.
        
    avg_state = {}
    
    # Get all keys from the first model state
    for key in states[0].keys():
        # Sum up the weighted tensors from all clients for the current key
        avg_state[key] = sum(state[key] * (weight / total_weight) for state, weight in zip(states, weights))
        
    return avg_state

# ==========================================================
# 1. LOAD DATASETS
# ==========================================================
datasets = {}
paths = {
    "HeartVitals": "data/real/Health_data.csv",
    "MITBIH": "datasets/health_status/kaggle_heartbeat/mitbih_train.csv", # FIXED: Using train set
    "Arrhythmia": "datasets/health_status/arrhythmia/arrhythmia.data" # Using .data file
}

# HeartVitals (3-class)
X_hv, y_hv = prep_hearvitals(paths["HeartVitals"])
if X_hv is not None:
    datasets["HeartVitals"] = (X_hv, y_hv)

# MITBIH (map 0..4 → 3-class)
X_m, y_m = load_mitbih(paths["MITBIH"])
if X_m is not None:
    datasets["MITBIH"] = (X_m, y_m)

# Arrhythmia (held-out)
heldout_name = "Arrhythmia"
X_a, y_a = load_arrhythmia(paths["Arrhythmia"])
if X_a is not None:
    datasets[heldout_name] = (X_a, y_a)
else:
    heldout_name = None
    print("Warning: Arrhythmia dataset not found. No hold-out test will be performed.")

# ==========================================================
# 2. PRE-PROCESSING (PADDING & SCALING)
# ==========================================================
train_clients = {k: v for k, v in datasets.items() if k != heldout_name and v[0] is not None}

if not train_clients:
    print("Error: No training data found. Exiting.")
    exit()

print("\n🔍 Training clients:")
max_features = 0
for k, (X, y) in train_clients.items():
    if X.shape[1] > max_features:
        max_features = X.shape[1]
    uniq, cnt = np.unique(y, return_counts=True)
    print(f" - {k:10s} | n={len(X):6d} | features={X.shape[1]} | class dist={dict(zip(uniq, cnt))}")

if heldout_name and datasets.get(heldout_name):
    Xh, yh = datasets[heldout_name]
    if Xh.shape[1] > max_features:
        max_features = Xh.shape[1]
    uniq, cnt = np.unique(yh, return_counts=True)
    print(f"🧪 Held-out test: {heldout_name} | n={len(Xh):6d} | features={Xh.shape[1]} | class dist={dict(zip(uniq, cnt))}")

D_MAX = max_features
print(f"\n📐 Unified input dimension across all datasets: {D_MAX}")

# Pad all datasets
for k in datasets.keys():
    if datasets[k][0] is not None:
        X, y = datasets[k]
        datasets[k] = (pad_to_max_dim(X, D_MAX), y)

# Separate train/test again after padding
train_clients_final = {k: v for k, v in datasets.items() if k != heldout_name and v[0] is not None}
X_held, y_held = datasets.get(heldout_name, (None, None))

# Create and fit the scaler ONLY on the training data
all_train_X = np.concatenate([X for X, y in train_clients_final.values()], axis=0)
scaler = StandardScaler().fit(all_train_X)
print(f"Scaler fitted on {len(all_train_X)} training samples.")

# Scale the held-out test set
if X_held is not None:
    X_held_scaled = scaler.transform(X_held)

# ==========================================================
# 3. FEDERATED TRAINING
# ==========================================================
NUM_CLASSES = 3 # We are mapping both training sets to 3 classes
global_model = MLP(D_MAX, NUM_CLASSES).to(DEVICE)
print(f"\n🚀 Cross-domain Federated Training (train on {list(train_clients_final.keys())})")

hist = {"rounds": [], "global_train_acc": [], "global_val_acc": [], "global_loss": [],
        "per_client": {}}

# Initialize per-client history
for name in train_clients_final.keys():
    hist["per_client"][name] = []

for rnd in range(1, ROUNDS + 1):
    local_states, tr_accs, val_accs, losses, weights = [], [], [], [], []
    per_client_round_data = {}

    print(f"----- Round {rnd:02d} -----")
    for name, (X, y) in train_clients_final.items():
        
        # stratified split; if a class has 1 sample, fallback to non-stratified
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42+rnd)
        except ValueError:
            X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42+rnd)

        # Client-Skipping Logic
        if len(np.unique(y_tr)) <= 1:
            print(f"  ⚠️ {name:10s} | SKIPPING: Only 1 class found in train split.")
            per_client_round_data[name] = {"train_acc": None, "val_acc": None, "loss": None}
            continue

        # Scale data *after* splitting
        X_tr_scaled = scaler.transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        model = MLP(D_MAX, NUM_CLASSES)
        model.load_state_dict(global_model.state_dict())
        
        # Train locally
        state, tr_acc, va_acc, loss = train_local(model, X_tr_scaled, y_tr, X_va_scaled, y_va, epochs=EPOCHS, lr=LR)

        # Store results
        local_states.append(state)
        tr_accs.append(tr_acc)
        val_accs.append(va_acc)
        losses.append(loss)
        weights.append(len(X_tr))
        
        hist["per_client"][name].append({"round": rnd, "train_acc": tr_acc, "val_acc": va_acc, "loss": loss})
        print(f"  {name:10s} | Train_Acc={tr_acc:.4f} | Val_Acc={va_acc:.4f} | Loss={loss:.4f}")

    hist["per_client"][rnd] = per_client_round_data 

    if local_states: 
        global_state = fedavg(local_states, weights)
        global_model.load_state_dict(global_state)

        # Average only the clients that actually trained
        gtr = float(np.average(tr_accs, weights=weights)) if tr_accs else 0.0
        gva = float(np.average(val_accs, weights=weights)) if val_accs else 0.0
        gls = float(np.mean(losses)) if losses else 0.0
    else:
        print("  ⚠️ WARNING: No clients trained this round. Global model not updated.")
        if hist["rounds"]:
             gtr, gva, gls = hist["global_train_acc"][-1], hist["global_val_acc"][-1], hist["global_loss"][-1]
        else:
             gtr, gva, gls = 0.0, 0.0, 0.0

    hist["rounds"].append(rnd)
    hist["global_train_acc"].append(gtr)
    hist["global_val_acc"].append(gva)
    hist["global_loss"].append(gls)

    print(f"  ➜ Global Train_Acc={gtr:.4f} | Global Val_Acc={gva:.4f} | Avg Loss={gls:.4f}")

# ==========================================================
# 4. HELD-OUT CROSS-DOMAIN TEST
# ==========================================================
if X_held is not None and y_held is not None:
    print("\n================ HELD-OUT CROSS-DOMAIN TEST ================")
    global_model.eval()
    
    with torch.no_grad():
        X_held_tensor = torch.tensor(X_held_scaled).to(DEVICE)
        preds = torch.argmax(global_model(X_held_tensor), dim=1).cpu().numpy()
        
        # Map the Arrhythmia dataset's 1–16 labels to the model's 0–2 labels
        arrhythmia_map = {
            1: 0,  # Normal
            2: 1,  # Supraventricular (Moderate)
            3: 1,  # Ventricular (Moderate)
            4: 2,  # Fusion (Critical)
            5: 2,  # Unknown (Critical)
            6: 1,  # Moderate
            7: 1,  # Moderate
            8: 1,  # Moderate
            9: 1,  # Moderate
            10: 1, # Moderate
            14: 1, # Moderate
            15: 1, # Moderate
            16: 1  # Other (treating 16 as abnormal/moderate)
        }

    # Filter and map held-out labels
    valid_indices = np.isin(y_held, list(arrhythmia_map.keys()))
    y_held_filtered = y_held[valid_indices]
    preds_filtered = preds[valid_indices]

    if len(y_held_filtered) > 0:
        y_held_mapped = np.array([arrhythmia_map.get(y) for y in y_held_filtered])
        target_names = ["Normal (0)", "Moderate (1)", "Critical (2)"]

        cm_ho = confusion_matrix(y_held_mapped, preds_filtered, labels=[0, 1, 2])
        report_text = classification_report(y_held_mapped, preds_filtered, labels=[0, 1, 2], target_names=target_names, zero_division=0)
        report_dict = classification_report(y_held_mapped, preds_filtered, labels=[0, 1, 2], target_names=target_names, output_dict=True, zero_division=0)
        acc_ho_filtered = accuracy_score(y_held_mapped, preds_filtered)
        
        print(f"🧪 Held-out Test on {heldout_name} Acc (Mapped to classes 0, 1, 2): {acc_ho_filtered:.4f}")
        print("Confusion Matrix (Mapped):\n", cm_ho)
        print("Classification Report (Filtered):\n", report_text)
        
        np.save(os.path.join(SAVE_DIR, "confusion_matrix_crossdomain.npy"), cm_ho)
        with open(os.path.join(SAVE_DIR, "classification_report_crossdomain.json"), "w") as f:
            json.dump(report_dict, f, indent=2)
    else:
        print(f"Model trained on classes [0, 1, 2], but the '{heldout_name}' dataset contains no overlapping labels.")
        np.save(os.path.join(SAVE_DIR, "confusion_matrix_crossdomain.npy"), np.zeros((3,3)))
        with open(os.path.join(SAVE_DIR, "classification_report_crossdomain.json"), "w") as f:
            json.dump({"note": "Arrhythmia test skipped. No overlapping classes found."}, f, indent=2)

# ==========================================================
# 5. SAVE FINAL MODEL & HISTORY
# ==========================================================
torch.save(global_model.state_dict(), os.path.join(SAVE_DIR, "global_health_risk_model_crossdomain.pt"))
with open(os.path.join(SAVE_DIR, "training_history_crossdomain.json"), "w") as f:
    json.dump(hist, f, indent=2, ensure_ascii=False)

print("\n✅ Cross-domain federated training complete.")
print(f"Results saved in '{SAVE_DIR}' folder.")
