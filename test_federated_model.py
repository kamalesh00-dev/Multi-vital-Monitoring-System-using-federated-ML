# ==========================================================
# 🧪 test_federated_model.py
# Evaluate trained Federated Health Risk Model on all datasets
# ==========================================================
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================================
# CONFIG
# ==========================================================
SAVE_DIR = "saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(SAVE_DIR, "global_health_risk_model_crossdomain.pt")
FINE_TUNED_PATH = os.path.join(SAVE_DIR, "global_health_risk_model_finetuned_arrhythmia.pt")

DATASETS = {
    "HealthVitals": "data/real/Health_data.csv",
    "MITBIH": "datasets/health_status/kaggle_heartbeat/mitbih_train.csv",
    "Arrhythmia": "datasets/health_status/arrhythmia/arrhythmia.data"
}

# ==========================================================
# MODEL
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
# DATA LOADERS
# ==========================================================
def load_healthvitals(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if "Status" not in df.columns:
        raise ValueError("❌ Missing 'Status' column in Health_data.csv")
    y = LabelEncoder().fit_transform(df["Status"])
    X = df.drop(columns=["Status"], errors="ignore").select_dtypes(include=[np.number]).values
    return X.astype(np.float32), y.astype(np.int64)


def load_mitbih(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].astype(int).values
    mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}
    y = np.array([mapping.get(int(v), 1) for v in y_raw], dtype=np.int64)
    return X, y


def load_arrhythmia(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values.astype(np.float32)
    y = df.iloc[:, -1].astype(int).values
    mapping = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1,
               10: 1, 14: 1, 15: 1, 16: 1}
    y_mapped = np.array([mapping.get(lbl, 1) for lbl in y], dtype=np.int64)
    return X, y_mapped


def pad_to_dim(X, D):
    if X.shape[1] == D:
        return X
    elif X.shape[1] < D:
        Z = np.zeros((X.shape[0], D), dtype=np.float32)
        Z[:, :X.shape[1]] = X
        return Z
    else:
        return X[:, :D]


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================
def evaluate_model(model_path, datasets):
    results = {}
    print(f"\n🚀 Loading model from: {model_path}")
    D_MAX = 274  # fixed to the training dimension

    model = MLP(D_MAX, 3).to(DEVICE)
    # --- Safe loader ---
    pretrained_dict = torch.load(model_path, map_location=DEVICE)
    model_dict = model.state_dict()
    compatible = {k: v for k, v in pretrained_dict.items() if k in model_dict and            model_dict[k].shape == v.shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    print(f"✅ Loaded {len(compatible)} compatible layers (ignored mismatched layers).")

    model.eval()

    for name, path in datasets.items():
        print(f"\n📂 Evaluating on: {name}")

        # --- Load dataset ---
        if "Health" in name:
            X, y = load_healthvitals(path)
        elif "MITBIH" in name:
            X, y = load_mitbih(path)
        elif "Arrhythmia" in name:
            X, y = load_arrhythmia(path)
        else:
            continue

        # --- Pad & scale ---
        X_padded = pad_to_dim(X, D_MAX)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_padded)

        # --- Predict ---
        with torch.no_grad():
            preds = torch.argmax(model(torch.tensor(X_scaled).to(DEVICE)), dim=1).cpu().numpy()

        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds, labels=[0, 1, 2])
        report = classification_report(
            y, preds,
            target_names=["Normal", "Moderate", "Critical"],
            zero_division=0,
            output_dict=True
        )

        results[name] = {
            "Accuracy": acc,
            "ConfusionMatrix": cm.tolist(),
            "Report": report
        }

        print(f"✅ Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", json.dumps(report, indent=2))

    return results


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n==============================================")
    print("🏥 FEDERATED HEALTH MODEL - EVALUATION REPORT")
    print("==============================================")

    # --- Evaluate global model ---
    print("\n🌍 Evaluating GLOBAL model...")
    global_results = evaluate_model(MODEL_PATH, DATASETS)

    # --- Evaluate fine-tuned model ---
    print("\n🧬 Evaluating FINE-TUNED model...")
    finetuned_results = evaluate_model(FINE_TUNED_PATH, DATASETS)

    # --- Save results to JSON ---
    output = {
        "GlobalModel": global_results,
        "FineTunedModel": finetuned_results
    }
    with open(os.path.join(SAVE_DIR, "evaluation_report_all_datasets.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Evaluation report saved to {SAVE_DIR}/evaluation_report_all_datasets.json")
    print("\n✅ Done.")
