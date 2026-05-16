import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# ==========================================
# 1. Find & Load the Data
# ==========================================
def load_data_binary():
    print("🔍 Searching for data...")
    
    # We look for the file you successfully found before
    possible_paths = [
        r".\datasets\health_status\arrhythmia\arrhythmia_clean.csv", # The one that worked!
        "arrhythmia_clean.csv",
        "datasets/arrhythmia_clean.csv"
    ]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Load carefully
                df = pd.read_csv(path, engine='python')
                # Check if it has real data (columns > 10)
                if df.shape[1] > 10:
                    print(f"✅ Loaded: {path} | Shape: {df.shape}")
                    break
            except:
                continue
    
    if df is None:
        print("❌ Error: Could not load data. Make sure 'arrhythmia_clean.csv' is in the folder.")
        sys.exit(1)

    # Clean missing values
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # Features (X) and Target (y)
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    # --- THE MAGIC FIX: Convert 13 Classes -> 2 Classes ---
    # Class 1 is 'Normal' in MIT-BIH standard.
    # We set: Normal (1) -> 0, Everything else -> 1
    # This makes the task much easier for the model.
    y = np.where(y_raw == 1, 0, 1)
    
    print(f"ℹ️  Binary Balance: {np.sum(y==0)} Healthy, {np.sum(y==1)} Arrhythmia")
    
    return X, y

# ==========================================
# 2. Train the Model
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # Load
    X, y = load_data_binary()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # To Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Model (Simple & Strong)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2) # Binary Output
    ).to(device)

    # Weighted Loss (To handle imbalance)
    class_counts = torch.bincount(y_train)
    weights = 1. / class_counts.float()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Loop
    print("\nStarting Training...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Final Report
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        print("\n✅ Final Accuracy Report:")
        print(classification_report(y_test.cpu(), preds.cpu(), target_names=["Healthy", "Arrhythmia"]))

if __name__ == "__main__":
    train()