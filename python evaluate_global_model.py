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
HEALTH_DATA_PATH = "data/real/Health_data.csv"
MITBIH_PATH = "datasets/health_status/kaggle_heartbeat/mitbih_train.csv"
ARRHYTHMIA_PATH = "datasets/health_status/arrhythmia/arrhythmia.data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# LOAD MODEL
# ==========================================================
D_MAX = X_test.shape[1]
model = MLP(D_MAX, 3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==========================================================
# SCALING (based on test itself for now)
# ==========================================================
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# ==========================================================
# PREDICTION
# ==========================================================
X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    preds = torch.argmax(model(X_tensor), dim=1).cpu().numpy()

# ==========================================================
# EVALUATION
# ==========================================================
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds, zero_division=0, output_dict=True)

print(f"\n🧪 Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(json.dumps(report, indent=2))

# ==========================================================
# VISUALIZATION
# ==========================================================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix (Accuracy={acc:.2f})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
