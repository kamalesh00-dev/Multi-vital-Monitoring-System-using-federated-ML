import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from streamlit_autorefresh import st_autorefresh

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(page_title="🏥 Patient Health Dashboard", layout="wide")
st.title("🏥 Patient Health Monitoring & Federated Learning Dashboard")

SAVE_DIR = "saved_models"

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
REFRESH_INTERVAL = st.sidebar.slider("Auto-Refresh Interval (seconds)", 5, 60, 10)
if auto_refresh:
    st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="datarefresh")

if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

# ======================================================
# 1️⃣ MODEL PERFORMANCE OVERVIEW
# ======================================================
st.header("📈 Model Performance Overview")

history = {}
try:
    with open(os.path.join(SAVE_DIR, "training_history.json"), "r") as f:
        history = json.load(f)

    rounds = history.get("rounds", [])
    accs = history.get("accuracy", [])
    losses = history.get("loss", [])

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(rounds, accs, marker="o", color="green")
        ax1.set_xlabel("Rounds")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Global Accuracy Trend")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(rounds, losses, marker="x", color="red")
        ax2.set_xlabel("Rounds")
        ax2.set_ylabel("Loss")
        ax2.set_title("Global Loss Trend")
        st.pyplot(fig2)

except Exception as e:
    st.warning(f"⚠️ Unable to load history: {e}")

# ======================================================
# 2️⃣ TRAINING HISTORY TABLE
# ======================================================
st.header("🧠 Training History Summary")
if history and "rounds" in history:
    df_history = pd.DataFrame({
        "Round": history.get("rounds", []),
        "Accuracy": history.get("accuracy", []),
        "Loss": history.get("loss", [])
    })
    st.dataframe(df_history, width='stretch')
else:
    st.info("Training history not available yet.")

# ======================================================
# 3️⃣ CONFUSION MATRIX
# ======================================================
st.header("📊 Confusion Matrix")

cm_path = os.path.join(SAVE_DIR, "confusion_matrix.npy")
if os.path.exists(cm_path):
    cm = np.load(cm_path)
    if np.issubdtype(cm.dtype, np.number):
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title("Confusion Matrix (Latest Round)")
        st.pyplot(fig_cm)
    else:
        st.error("Confusion matrix data invalid format.")
else:
    st.info("Confusion matrix not available yet. Run training first.")

# ======================================================
# 4️⃣ CLIENT PERFORMANCE
# ======================================================
st.header("🏥 Per-Client Accuracy Trends")

try:
    if "clients" in history:
        fig_client, ax_client = plt.subplots()
        for client, data in history["clients"].items():
            ax_client.plot(range(1, len(data["accuracy"]) + 1),
                           data["accuracy"], marker="o", label=client)
        ax_client.legend()
        ax_client.set_xlabel("Rounds")
        ax_client.set_ylabel("Accuracy")
        ax_client.set_title("Client Accuracy Comparison")
        st.pyplot(fig_client)
    else:
        st.info("No client data found.")
except Exception:
    st.info("Client accuracy unavailable.")

# ======================================================
# 5️⃣ LIVE ECG SIMULATION
# ======================================================
st.header("🫀 Live ECG Simulation")

fs = 200
duration = 3
t = np.linspace(0, duration, fs * duration)
signal = 0.6 * np.sin(2 * np.pi * 1.7 * t) + 0.3 * np.sin(2 * np.pi * 3.5 * t)
signal += np.random.normal(0, 0.05, len(t))
signal += np.exp(-((t - 1.2) * 10) ** 2) * 2.2

fig_ecg, ax_ecg = plt.subplots(figsize=(8, 3))
ax_ecg.plot(t, signal, color="crimson")
ax_ecg.set_ylim([-2, 3])
ax_ecg.set_title("Simulated ECG Signal (Live)")
ax_ecg.set_xlabel("Time (s)")
ax_ecg.set_ylabel("Amplitude (mV)")
st.pyplot(fig_ecg)

# ======================================================
# 6️⃣ LIVE TIME-SERIES VITAL SIGN ANALYSIS (IEEE STYLE)
# ======================================================
from utils.vitals_plot import plot_vitals   # ✅ import the vitals plot function

st.header("📊 Live Time-Series Vital Sign Analysis")

# Generate and show the plot
fig_ts = plot_vitals()
st.pyplot(fig_ts)



axs[0, 0].plot(minutes, SBP, label="SBP", color='blue', marker='o')
axs[0, 0].plot(minutes, DBP, label="DBP", color='red', marker='^')
axs[0, 0].set_title("(a) Systolic & Diastolic BP")
axs[0, 0].set_xlabel("Time (min)")
axs[0, 0].set_ylabel("mmHg")
axs[0, 0].legend()

axs[1, 0].plot(minutes, HR, label="HR", color='maroon', marker='x')
axs[1, 0].set_title("(b) Heart Rate")
axs[1, 0].set_xlabel("Time (min)")
axs[1, 0].set_ylabel("bpm")

axs[1, 1].plot(minutes, Temp, label="Temp", color='darkcyan', marker='s')
axs[1, 1].set_title("(c) Body Temperature")
axs[1, 1].set_xlabel("Time (min)")
axs[1, 1].set_ylabel("°C")

plt.tight_layout()
st.pyplot(fig_ts)
# ======================================================
# 6️⃣ LIVE TIME-SERIES VITAL SIGN ANALYSIS (IEEE STYLE)
# ======================================================
from utils.vitals_plot import plot_vitals   # ✅ import the vitals plot function

st.header("📊 Live Time-Series Vital Sign Analysis")

# Generate and show the plot
fig_ts = plot_vitals()
st.pyplot(fig_ts)


# ======================================================
# 8️⃣ PATIENT DETAILS
# ======================================================
st.header("🧍 Patient Details")

sample_patients = pd.DataFrame({
    "Patient ID": ["P001", "P002", "P003", "P004"],
    "Age": [45, 60, 39, 52],
    "Gender": ["M", "F", "M", "F"],
    "Pulse (BPM)": [88, 102, 95, 76],
    "Temperature (°C)": [36.8, 37.6, 37.2, 36.9],
    "SpO₂ (%)": [98, 95, 97, 99],
    "Status": ["Stable", "Alert", "Stable", "Stable"]
})
st.dataframe(sample_patients, width='stretch')


# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("Developed for Federated Health Monitoring Project © 2025")
