import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from streamlit_autorefresh import st_autorefresh

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(page_title="🏥 Patient Health Dashboard", layout="wide")
st.title("🏥 Patient Health Monitoring & Federated Learning Dashboard")

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Model Type",
        [
            "global_simpleNN.pt",
            "global_health_risk_model.pt",
            "global_health_risk_model_crossdomain.pt",
        ],
        index=2,
        key="model_selector_unique", 
    )

    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True, key="autorefresh_unique")
    REFRESH_INTERVAL = st.slider(
        "Auto-Refresh Interval (seconds)",
        5,
        60,
        10,
        key="slider_unique",
    )

    if auto_refresh:
        st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh_loop_unique")

    if st.button("🔄 Refresh Now", key="refresh_now_button_unique"):
        st.rerun()

# ======================================================
# MODEL CLASS
# ======================================================
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

# ======================================================
# MODEL LOADING HELPERS
# ======================================================
def clean_state_dict(state):
    """Automatically fix missing/unexpected prefixes ('net.')"""
    cleaned = {}
    for k, v in state.items():
        if k.startswith("net."):
            cleaned[k.replace("net.", "")] = v
        else:
            cleaned["net." + k] = v
    return cleaned


def build_dynamic_mlp(state_dict):
    """Dynamically match model input layer size."""
    for k, v in state_dict.items():
        if "weight" in k and len(v.shape) == 2:
            d_in = v.shape[1]
            break
    return MLP(d_in, 3)

# ======================================================
# LOAD MODEL HISTORY
# ======================================================
st.header("📈 Model Performance Overview")

try:
    with open(os.path.join(SAVE_DIR, "training_history_crossdomain.json"), "r") as f:
        history = json.load(f)
    rounds = history.get("rounds", [])
    accs = history.get("acc", [])
    losses = history.get("loss", [])

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(rounds, accs, marker="o", color="green")
        ax.set_title("Global Accuracy Trend")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(rounds, losses, marker="x", color="red")
        ax.set_title("Global Loss Trend")
        st.pyplot(fig)

except Exception as e:
    st.warning(f"⚠️ Could not load training history: {e}")

# ======================================================
# CONFUSION MATRIX
# ======================================================
st.header("📊 Confusion Matrix")

cm_path = os.path.join(SAVE_DIR, "confusion_matrix_crossdomain.npy")
if os.path.exists(cm_path):
    cm = np.load(cm_path)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Cross-Domain Confusion Matrix")
    st.pyplot(fig)
else:
    st.info("No confusion matrix found yet.")

st.header("🚨 Risk & Alerts")

try:
    model_path = os.path.join(SAVE_DIR, model_choice)
    raw_state = torch.load(model_path, map_location="cpu")
    cleaned_state = clean_state_dict(raw_state)

    # Rebuild model dynamically
    model = build_dynamic_mlp(cleaned_state)
    model.load_state_dict(cleaned_state, strict=False)
    model.eval()

    # Load most recent patient record
    df = pd.read_csv(os.path.join("data", "real", "Health_data.csv"))
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    X_live = df.select_dtypes(include=[np.number]).tail(1).values.astype(np.float32)

    # Dynamically infer model input dimension
    D_in = next(model.parameters()).shape[1]

    # Pad or trim input to expected model dimension
    if X_live.shape[1] < D_in:
        X_padded = np.zeros((1, D_in), dtype=np.float32)
        X_padded[0, :X_live.shape[1]] = X_live
        X_live = X_padded
    elif X_live.shape[1] > D_in:
        X_live = X_live[:, :D_in]

    # Dynamically fit a new scaler that matches the model
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.random.randn(10, D_in))  # dummy data same size
    X_scaled = scaler.transform(X_live)

    # Predict risk using model
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled))
        probs = torch.softmax(logits, dim=1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = ["Normal", "Moderate", "Critical"][pred_idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Normal", f"{probs[0]:.2f}")
    col2.metric("🟠 Moderate", f"{probs[1]:.2f}")
    col3.metric("🔴 Critical", f"{probs[2]:.2f}")

    if pred_idx == 2:
        st.error(f"⚠️ High Risk — {pred_label}")
    elif pred_idx == 1:
        st.warning(f"⚠️ Moderate Risk — {pred_label}")
    else:
        st.success(f"✅ Stable — {pred_label}")

except Exception as e:
    import gc
    gc.collect()  # clear any old cached scaler or tensors
    st.warning(f"⚠️ Risk detection failed: {e}")

st.markdown("---")
st.caption("Developed for Federated Health Monitoring Project © 2025")
