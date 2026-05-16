import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ======================================================
# 1️⃣ PATHS
# ======================================================
DATA_DIR = os.path.join("datasets", "health_status", "kaggle_heartbeat")
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# 2️⃣ LOAD DATA
# ======================================================
train = pd.read_csv(os.path.join(DATA_DIR, "mitbih_train.csv"), header=None).values
test  = pd.read_csv(os.path.join(DATA_DIR, "mitbih_test.csv"), header=None).values

X_train, y_train = train[:, :-1].astype(np.float32), train[:, -1].astype(int)
X_test,  y_test  = test[:,  :-1].astype(np.float32), test[:,  -1].astype(int)

# Normalize each sample (zero mean, unit variance)
X_train = (X_train - X_train.mean(1, keepdims=True)) / (X_train.std(1, keepdims=True) + 1e-6)
X_test  = (X_test  - X_test.mean(1, keepdims=True))  / (X_test.std(1, keepdims=True) + 1e-6)

# ======================================================
# 3️⃣ TORCH DATASETS
# ======================================================
def to_loader(X, y, bs=256, shuffle=True):
    ds = torch.utils.data.TensorDataset(torch.tensor(X)[:, None, :], torch.tensor(y))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle)

train_loader = to_loader(X_train, y_train)
test_loader  = to_loader(X_test, y_test, shuffle=False)

# ======================================================
# 4️⃣ MODEL DEFINITION
# ======================================================
class ECG1DCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.fe(x))

model = ECG1DCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ======================================================
# 5️⃣ EVALUATION FUNCTION
# ======================================================
def eval_acc():
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds.extend(model(xb).argmax(1).cpu().numpy())
            ys.extend(yb.numpy())
    return accuracy_score(ys, preds)

# ======================================================
# 6️⃣ TRAIN LOOP
# ======================================================
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1:02d} | Test Acc: {eval_acc():.4f}")

# ======================================================
# 7️⃣ SAVE MODEL
# ======================================================
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ecg_cnn.pt"))
print("\n✅ ECG model saved ➜ saved_models/ecg_cnn.pt")
