import os, json, math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# ===========================
# Paths (auto-detect)
# ===========================
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

POSSIBLE_HEART_FILES = [
    os.path.join("data","real","Health_data.csv"),
    os.path.join("datasets","health_status","Health_data.csv"),
    os.path.join("datasets","Health_data.csv"),
]
POSSIBLE_ARR_FILES = [
    os.path.join("datasets","health_status","arrhythmia","arrhythmia.data"),
    os.path.join("datasets","arrhythmia","arrhythmia.data"),
    os.path.join("data","real","arrhythmia.data"),
]
POSSIBLE_MIT_FILES = [
    os.path.join("datasets","health_status","kaggle_heartbeat","mitbih_train.csv"),
    os.path.join("datasets","kaggle_heartbeat","mitbih_train.csv"),
    os.path.join("data","real","mitbih_train.csv"),
]

def pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

HEART_PATH = pick_first_existing(POSSIBLE_HEART_FILES)
ARR_PATH   = pick_first_existing(POSSIBLE_ARR_FILES)
MIT_PATH   = pick_first_existing(POSSIBLE_MIT_FILES)

if HEART_PATH is None:
    raise FileNotFoundError("Heart disease/vitals dataset not found. Place Health_data.csv under data/real/ or datasets/health_status/")

if ARR_PATH is None:
    print("⚠️ Arrhythmia dataset not found — continuing with other clients.")
if MIT_PATH is None:
    print("⚠️ MIT-BIH heartbeat dataset not found — continuing with other clients.")

# ===========================
# Label space and mapping
# Target: 0 Normal, 1 Moderate, 2 Critical
# ===========================
RISK_LABELS = ["Normal","Moderate","Critical"]

def heart_status_to_risk(y):
    # Heart-disease style labels: 0 = no disease, >0 = disease
    # If you already have Status as strings, map them here if needed
    y = np.asarray(y)
    out = np.zeros_like(y, dtype=int)
    out[y > 0] = 2  # make disease => Critical (you can tune to 1 if you want softer mapping)
    return out

def arrhythmia_to_risk(y):
    # UCI arrhythmia: class 1 = normal, 2..16 = arrhythmia types
    # Map ventricular-like (severe) to Critical, others to Moderate
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y, dtype=int)
    # invalid to 0 first
    out[:] = 0
    # known moderate (most arrhythmias except severe ventricular)
    moderate = [2,3,4,5,6,7,8,9,10,13,14,15,16]
    # severe/critical (PVC/Ventricular etc.) rough grouping
    critical = [11,12]
    out[np.isin(y, moderate)] = 1
    out[np.isin(y, critical)] = 2
    out[y == 1] = 0
    return out

def mitbih_to_risk(y):
    # Kaggle MIT-BIH beat classes:
    # 0 = N (normal), 1 = S, 2 = V, 3 = F, 4 = Q
    # Map: 0 -> Normal, 1,3 -> Moderate, 2,4 -> Critical
    y = np.asarray(y, dtype=int)
    out = np.zeros_like(y, dtype=int)
    out[np.isin(y, [1,3])] = 1
    out[np.isin(y, [2,4])] = 2
    return out

# ===========================
# MIT-BIH feature extraction (1D signal → compact features)
# ===========================
def ecg_features_187(x):
    # x shape: (187,)
    x = x.astype(np.float32)
    # normalize per beat
    x = (x - x.mean())/(x.std()+1e-6)
    feats = []
    # time features
    feats.append(x.mean())                        # ~0 after norm
    feats.append(x.std())
    feats.append(np.max(x)-np.min(x))            # ptp
    feats.append(np.max(np.abs(x)))              # max abs
    feats.append(np.median(x))
    # energy
    feats.append(float(np.sum(x**2)))
    # simple slopes
    dx = np.diff(x)
    feats.append(np.mean(dx))
    feats.append(np.std(dx))
    # frequency features
    fft = np.fft.rfft(x)
    mag = np.abs(fft)
    # spectral centroid
    freqs = np.linspace(0, 1, len(mag))
    sc = float(np.sum(freqs*mag)/(np.sum(mag)+1e-6))
    feats.append(sc)
    # top-3 spectral peaks
    top3 = np.sort(mag)[-3:]
    feats.extend(list(top3))
    return np.array(feats, dtype=np.float32)

# ===========================
# Loaders
# ===========================
def load_heart_dataset(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ","_") for c in df.columns]
    if "Status" in df.columns:
        # If Status numeric/string: map to risk
        if df["Status"].dtype.kind in "iu":
            y = heart_status_to_risk(df["Status"].values)
            X = df.drop(columns=["Status"])
        else:
            # String labels — map flexibly
            lab = df["Status"].str.lower().str.strip()
            # basic mapping; tune if you like
            normal = lab.isin(["normal","healthy","0"])
            critical = lab.isin(["disease","risk","critical","1","2"])
            y = np.zeros(len(df), dtype=int)
            y[critical.values] = 2
            X = df.drop(columns=["Status"])
    else:
        raise ValueError("Expected 'Status' column in Health_data.csv")
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf,-np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    return X.values.astype(np.float32), y.astype(int)

def load_arrhythmia_dataset(path):
    # UCI arrhythmia.data (279 attributes, last is class, '?' as missing)
    df = pd.read_csv(path, header=None, na_values=["?"])
    # Last col is class
    y = df.iloc[:,-1].values
    X = df.iloc[:,:-1]
    # impute numeric
    X = X.replace([np.inf,-np.inf], np.nan)
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(X))
    # risk map
    y = arrhythmia_to_risk(y)
    return X.values.astype(np.float32), y.astype(int)

def load_mitbih_dataset(path):
    train = pd.read_csv(path, header=None).values
    Xsig = train[:,:-1].astype(np.float32)
    y_raw = train[:,-1].astype(int)
    # extract features per beat
    feats = np.stack([ecg_features_187(x) for x in Xsig], axis=0)
    y = mitbih_to_risk(y_raw)
    return feats, y.astype(int)

# ===========================
# Build clients (one per dataset)
# ===========================
clients = []
client_names = []

X_heart, y_heart = load_heart_dataset(HEART_PATH)
clients.append(("HeartVitals", X_heart, y_heart)); client_names.append("HeartVitals")

if ARR_PATH is not None:
    try:
        X_arr, y_arr = load_arrhythmia_dataset(ARR_PATH)
        clients.append(("Arrhythmia", X_arr, y_arr)); client_names.append("Arrhythmia")
    except Exception as e:
        print(f"⚠️ Arrhythmia load failed: {e}")

if MIT_PATH is not None:
    try:
        X_mit, y_mit = load_mitbih_dataset(MIT_PATH)
        clients.append(("MITBIH", X_mit, y_mit)); client_names.append("MITBIH")
    except Exception as e:
        print(f"⚠️ MIT-BIH load failed: {e}")

if len(clients) == 0:
    raise RuntimeError("No clients assembled. Provide at least one dataset.")

print("\n🔍 Clients assembled:")
for name, Xc, yc in clients:
    counts = np.bincount(yc, minlength=3)
    print(f" - {name:10s} | samples={len(yc):5d} | class dist={counts.tolist()}")

# ===========================
# Train/Val split per client & scaling
# ===========================
def train_val_split(X, y, frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(len(X)*frac)
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

client_data = []
scalers = []
for (name, Xc, yc) in clients:
    Xtr, ytr, Xval, yval = train_val_split(Xc, yc, frac=0.2)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xval = scaler.transform(Xval).astype(np.float32)
    client_data.append((name, Xtr, ytr, Xval, yval))
    scalers.append(scaler)

# ===========================
# Model (shared)
# ===========================
class MLP(nn.Module):
    def __init__(self, d_in, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.net(x)

# unify input dim: use max feature size among clients (pad smaller with zeros)
in_dims = [c[1].shape[1] for c in client_data]
D = max(in_dims)
def pad_X(X, D):
    if X.shape[1] == D: return X
    Z = np.zeros((X.shape[0], D), dtype=np.float32)
    Z[:,:X.shape[1]] = X
    return Z

# ===========================
# Federated Training
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS=3; ROUNDS=20; BATCH=256; LR=1e-3
criterion = nn.CrossEntropyLoss()

def to_loader(X, y, bs=BATCH, shuffle=True):
    ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle)

def client_train(global_state, X, y):
    model = MLP(D, 3).to(DEVICE)
    model.load_state_dict(global_state)
    opt = optim.Adam(model.parameters(), lr=LR)
    model.train()
    loader = to_loader(X, y)
    last_loss = 0.0
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
    # return cpu weights
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, last_loss

def evaluate(model_state, X, y):
    model = MLP(D, 3).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X).to(DEVICE)).cpu().numpy()
    preds = logits.argmax(1)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds, labels=[0,1,2])
    return acc, cm

# init global
global_model = MLP(D, 3).to(DEVICE)
global_state = {k:v.detach().cpu().clone() for k,v in global_model.state_dict().items()}

history = {"rounds": [], "accuracy": [], "loss": [], "clients": {}}
for (name, *_ ) in client_data:
    history["clients"][name] = {"accuracy": []}
client_records = []

print("\n🚀 Starting Combined Federated Training ...\n")
for r in range(1, ROUNDS+1):
    local_states = []
    local_losses = []
    weights = []
    print(f"----- Round {r:02d} -----")
    # each client trains on its padded features
    for (name, Xtr, ytr, Xval, yval) in client_data:
        Xtr_pad = pad_X(Xtr, D)
        state_i, loss_i = client_train(global_state, Xtr_pad, ytr)
        local_states.append(state_i)
        local_losses.append(loss_i)
        weights.append(len(ytr))

        # client train-acc (on val set is better indicator)
        Xval_pad = pad_X(Xval, D)
        acc_i, _ = evaluate(state_i, Xval_pad, yval)
        history["clients"][name]["accuracy"].append(float(acc_i))
        client_records.append({"Round": r, "Client": name, "TrainSamples": int(len(ytr)), "LocalLoss": float(loss_i), "LocalAcc": float(acc_i)})
        print(f"  {name:10s} | n={len(ytr):5d} | ValAcc={acc_i:.4f} | Loss={loss_i:.4f}")

    # FedAvg weighted by samples
    total = float(sum(weights))
    new_state = {}
    for k in global_state.keys():
        agg = None
        for st, w in zip(local_states, weights):
            contrib = st[k]*(w/total)
            agg = contrib if agg is None else agg + contrib
        new_state[k] = agg
    global_state = new_state

    # Global evaluation on concatenated val sets
    Xall = []; yall = []
    for (_, _, _, Xval, yval) in client_data:
        Xall.append(pad_X(Xval, D)); yall.append(yval)
    Xall = np.vstack(Xall); yall = np.concatenate(yall)
    g_acc, g_cm = evaluate(global_state, Xall, yall)
    avg_loss = float(np.mean(local_losses))
    print(f"  ➜ Global Val Acc: {g_acc:.4f} | Avg Local Loss: {avg_loss:.4f}")

    history["rounds"].append(r); history["accuracy"].append(float(g_acc)); history["loss"].append(avg_loss)
    with open(os.path.join(SAVE_DIR, "training_history_combined.json"), "w") as f:
        json.dump(history, f, indent=2)
    np.save(os.path.join(SAVE_DIR, "confusion_matrix_combined.npy"), g_cm)

# Save final global model
torch.save(global_state, os.path.join(SAVE_DIR, "global_health_risk_model.pt"))

# Save per-client metrics
pd.DataFrame(client_records).to_csv(os.path.join(SAVE_DIR, "client_metrics_combined.csv"), index=False)

# Save label map
with open(os.path.join(SAVE_DIR, "risk_labels.json"), "w") as f:
    json.dump({"0":"Normal","1":"Moderate","2":"Critical"}, f)

print("\n✅ Training complete.")
print("Saved:")
print(" - saved_models/global_health_risk_model.pt (state_dict)")
print(" - saved_models/training_history_combined.json")
print(" - saved_models/confusion_matrix_combined.npy")
print(" - saved_models/client_metrics_combined.csv")
print(" - saved_models/risk_labels.json")
