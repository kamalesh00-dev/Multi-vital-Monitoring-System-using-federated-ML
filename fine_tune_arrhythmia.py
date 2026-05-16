import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import os

# ==========================================================
# CONFIG
# ==========================================================
SAVE_DIR = "saved_models"
MODEL_PATH = os.path.join(SAVE_DIR, "global_health_risk_model_crossdomain.pt")
ARRHYTHMIA_PATH = "datasets/health_status/arrhythmia/arrhythmia.data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
LR = 1e-4
BATCH = 64

# ==========================================================
# MODEL (same as before)
# ==========================================================
class MLP(nn.Module):
    def __init__(self, d_in, n_classes=3):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.BatchNorm1d(128),  # ✅ Added normalization for stability
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================================
# LOAD & MAP ARRHYTHMIA DATA
# ==========================================================
def load_arrhythmia(path):
    df = pd.read_csv(path, sep=",", on_bad_lines="skip", engine="python")
    y = df.iloc[:, -1].astype(int).values
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    return X.astype(np.float32), y.astype(np.int64)

X, y = load_arrhythmia(ARRHYTHMIA_PATH)

arrhythmia_map = {
    1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 14: 1, 15: 1, 16: 1
}
mask = np.isin(y, list(arrhythmia_map.keys()))
y = np.array([arrhythmia_map.get(int(v), 1) for v in y[mask]])
X = X[mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
D_IN = X_scaled.shape[1]

# ==========================================================
# LOAD GLOBAL MODEL
# ==========================================================
model = MLP(D_IN, 3).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
print("✅ Global model loaded. Starting fine-tuning...")

# ==========================================================
# TRAIN / VALID SPLIT
# ==========================================================
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_t, y_train_t),
                                           batch_size=BATCH, shuffle=True)

# ==========================================================
# TRAINING LOOP
# ==========================================================
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_val_t), dim=1)
        acc = accuracy_score(y_val, preds.cpu().numpy())
    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss={running_loss/len(train_loader):.4f} | Val_Acc={acc:.4f}")

# ==========================================================
# EVALUATE ON FULL SET
# ==========================================================
model.eval()
with torch.no_grad():
    preds = torch.argmax(model(torch.tensor(X_scaled).to(DEVICE)), dim=1).cpu().numpy()

acc = accuracy_score(y, preds)
cm = confusion_matrix(y, preds)
report = classification_report(y, preds, labels=[0,1,2],
                               target_names=["Normal (0)", "Moderate (1)", "Critical (2)"])

print("\n✅ Fine-tuning complete.")
print(f"🧪 Overall Accuracy (Fine-Tuned): {acc:.4f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", cm)

# Save fine-tuned model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "global_health_risk_model_finetuned_arrhythmia.pt"))
print(f"\n💾 Fine-tuned model saved to: {SAVE_DIR}/global_health_risk_model_finetuned_arrhythmia.pt")
