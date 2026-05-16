import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ==========================================
# 1. Load Data (Binary Mode)
# ==========================================
def load_data_binary():
    print("🔍 Searching for data...")
    possible_paths = [
        r".\datasets\health_status\arrhythmia\arrhythmia_clean.csv",
        "arrhythmia_clean.csv",
        "datasets/arrhythmia_clean.csv"
    ]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, engine='python')
                if df.shape[1] > 10:
                    print(f"✅ Loaded: {path} | Shape: {df.shape}")
                    break
            except:
                continue
    
    if df is None:
        print("❌ Error: Could not load data.")
        sys.exit(1)

    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    # Convert to Binary: 1=Normal (0), Others=Sick (1)
    y = np.where(y_raw == 1, 0, 1)
    
    print(f"ℹ️  Binary Balance: {np.sum(y==0)} Healthy, {np.sum(y==1)} Arrhythmia")
    return X, y

# ==========================================
# 2. Advanced Training
# ==========================================
def train():
   
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    X, y = load_data_binary()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # --- UPGRADED MODEL ARCHITECTURE ---
    # We added Dropout and more layers to handle the complexity
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),  # Bigger layer (was 64)
        nn.BatchNorm1d(128),               # Stabilizes learning
        nn.ReLU(),
        nn.Dropout(0.3),                   # Prevents overfitting
        
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(64, 2)                   # Output
    ).to(device)

    # Loss & Optimizer
    class_counts = torch.bincount(y_train)
    weights = 1. / class_counts.float()
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Weight decay adds L2 regularization (helps generalization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print("\nStarting Extended Training (300 Epochs)...")
    epochs = 300  # Increased from 50
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        print("\n✅ Final High-Accuracy Report:")
        print(classification_report(y_test.cpu(), preds.cpu(), target_names=["Healthy", "Arrhythmia"]))

if __name__ == "__main__":
    train()
