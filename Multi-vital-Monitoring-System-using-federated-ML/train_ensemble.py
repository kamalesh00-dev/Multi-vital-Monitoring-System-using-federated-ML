import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# ==========================================
# 1. Data Loading & Dual-Preprocessing
# ==========================================
def get_data_streams():
    print("🔍 Loading data...")
    path = r".\datasets\health_status\arrhythmia\arrhythmia_clean.csv"
    
    if not os.path.exists(path):
        for p in ["arrhythmia_clean.csv", "datasets/arrhythmia_clean.csv"]:
            if os.path.exists(p):
                path = p
                break

    df = pd.read_csv(path, engine='python')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    X_raw = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    y = np.where(y_raw == 1, 0, 1) # Binary
    
    print(f"   Original Data: {X_raw.shape}")

    # --- STREAM A: Feature Selection (For MLP) ---
    print("⚙️ Pipeline A: Random Forest Selection...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_raw, y)
    indices = np.argsort(rf.feature_importances_)[::-1][:50]
    X_mlp = X_raw[:, indices] # Top 50 features
    
    # --- STREAM B: PCA (For BiLSTM) ---
    print("⚙️ Pipeline B: PCA Compression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=64)
    X_lstm = pca.fit_transform(X_scaled) # 64 Components

    return X_mlp, X_lstm, y

# ==========================================
# 2. Define Models
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        # Input 1 feature per step, sequence length 64
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64*2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last step
        return self.fc(out)

# ==========================================
# 3. Train & Ensemble
# ==========================================
def train_ensemble():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu") # CPU is fine
    
    # 1. Get Data Streams
    X_mlp, X_lstm, y = get_data_streams()
    
    # Split (MUST use same seed to keep rows aligned)
    X_m_train, X_m_test, y_train, y_test = train_test_split(X_mlp, y, test_size=0.2, random_state=42)
    X_l_train, X_l_test, _, _ = train_test_split(X_lstm, y, test_size=0.2, random_state=42)
    
    # Scale MLP Data (LSTM PCA is already scaled)
    scaler = StandardScaler()
    X_m_train = scaler.fit_transform(X_m_train)
    X_m_test = scaler.transform(X_m_test)
    
    # Tensor Setup
    t_m_train = torch.tensor(X_m_train, dtype=torch.float32).to(device)
    t_m_test = torch.tensor(X_m_test, dtype=torch.float32).to(device)
    
    # Reshape LSTM data: (Batch, 64, 1)
    t_l_train = torch.tensor(X_l_train, dtype=torch.float32).unsqueeze(-1).to(device)
    t_l_test = torch.tensor(X_l_test, dtype=torch.float32).unsqueeze(-1).to(device)
    
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    
    # 2. Train Model A (MLP)
    print("\n🚀 Training Stream A (MLP)...")
    model_mlp = SimpleMLP().to(device)
    opt_m = optim.Adam(model_mlp.parameters(), lr=0.003, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    
    for epoch in range(60):
        model_mlp.train()
        opt_m.zero_grad()
        loss = crit(model_mlp(t_m_train), y_train_t)
        loss.backward()
        opt_m.step()
        
    # 3. Train Model B (BiLSTM)
    print("🚀 Training Stream B (BiLSTM)...")
    model_lstm = BiLSTM().to(device)
    opt_l = optim.Adam(model_lstm.parameters(), lr=0.002)
    
    for epoch in range(80):
        model_lstm.train()
        opt_l.zero_grad()
        loss = crit(model_lstm(t_l_train), y_train_t)
        loss.backward()
        opt_l.step()

    # 4. Ensemble Voting
    print("\n🤖 Calculating Ensemble Predictions...")
    model_mlp.eval()
    model_lstm.eval()
    
    with torch.no_grad():
        # Get Probabilities
        logits_m = model_mlp(t_m_test)
        probs_m = torch.softmax(logits_m, dim=1) # Shape (N, 2)
        
        logits_l = model_lstm(t_l_test)
        probs_l = torch.softmax(logits_l, dim=1) # Shape (N, 2)
        
        # WEIGHTED VOTE: Trust MLP (0.7) more than BiLSTM (0.3)
        final_probs = (0.7 * probs_m) + (0.3 * probs_l)
        final_preds = final_probs.argmax(dim=1)
        
    # 5. Report
    print("\n✅ Final Hybrid Ensemble Report:")
    print(classification_report(y_test, final_preds.numpy(), target_names=["Healthy", "Arrhythmia"]))

if __name__ == "__main__":
    train_ensemble()