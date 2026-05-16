import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================================
# CONFIG
# ==========================================================
SAVE_DIR = "saved_models"
MODEL_PATH = os.path.join(SAVE_DIR, "global_health_risk_model_crossdomain.pt")
ARRHYTHMIA_PATH = "datasets/health_status/arrhythmia/arrhythmia.data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================================
# MODEL (must match training architecture)
# ==========================================================
class MLP(nn.Module):
    def __init__(self, d_in, n_classes=3):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================================
# LOAD TEST DATA (Arrhythmia)
# ==========================================================
def load_arrhythmia(path):
    df = pd.read_csv(path, sep=",", on_bad_lines="skip", engine="python")
    y = df.iloc[:, -1].astype(int).values
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
    return X.astype(np.float32), y.astype(np.int64)

X_test, y_test = load_arrhythmia(ARRHYTHMIA_PATH)
print(f"Loaded Arrhythmia Test Set: {X_test.shape}")

# ==========================================================
# MAP LABELS (1–16 → 0,1,2)
# ==========================================================
arrhythmia_map = {
    1: 0,  # Normal
    2: 1,  # Moderate
    3: 1,  # Moderate
    4: 2,  # Critical
    5: 2,  # Critical
    6: 1,  # Moderate
    7: 1,  # Moderate
    8: 1,  # Moderate
    9: 1,  # Moderate
    10: 1, # Moderate
    14: 1, # Moderate
    15: 1, # Moderate
    16: 1  # Moderate
}

# Filter & map only valid classes
mask = np.isin(y_test, list(arrhythmia_map.keys()))
y_test_mapped = np.array([arrhythmia_map.get(int(v), 1) for v in y_test[mask]])
X_test = X_test[mask]
print(f"Valid samples after label mapping: {len(y_test_mapped)}")

# ==========================================================
# SCALE FEATURES
# ==========================================================
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# ==========================================================
# LOAD MODEL
# ==========================================================
D_MAX = X_test_scaled.shape[1]
model = MLP(D_MAX, 3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully.")

# ==========================================================
# PREDICTION
# ==========================================================
X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    preds = torch.argmax(model(X_tensor), dim=1).cpu().numpy()

unique, counts = np.unique(preds, return_counts=True)
print(f"Prediction distribution: {dict(zip(unique, counts))}")

# ==========================================================
# EVALUATION
# ==========================================================
acc = accuracy_score(y_test_mapped, preds)
cm = confusion_matrix(y_test_mapped, preds, labels=[0,1,2])
report = classification_report(y_test_mapped, preds, labels=[0,1,2],
                               target_names=["Normal (0)", "Moderate (1)", "Critical (2)"],
                               zero_division=0)

print(f"\n🧪 Test Accuracy (3-class mapped): {acc:.4f}")
print("\nClassification Report:\n", report)

# ==========================================================
# VISUALIZATION
# ==========================================================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0","Pred 1","Pred 2"], yticklabels=["True 0","True 1","True 2"])
plt.title(f"Confusion Matrix (Accuracy={acc:.2f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
