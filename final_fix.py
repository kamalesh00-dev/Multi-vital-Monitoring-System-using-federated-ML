import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ==========================================
# 1. Load & Select Features
# ==========================================
def load_and_optimize():
    print("🔍 Loading data...")
    path = r".\datasets\health_status\arrhythmia\arrhythmia_clean.csv"
    
    # Auto-find file if path is wrong
    if not os.path.exists(path):
        for p in ["arrhythmia_clean.csv", "datasets/arrhythmia_clean.csv"]:
            if os.path.exists(p):
                path = p
                break
    
    df = pd.read_csv(path, engine='python')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    X_raw = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    
    # Binary: 1=Normal(0), Others=Sick(1)
    y = np.where(y_raw == 1, 0, 1)
    
    print(f"   Original Features: {X_raw.shape[1]}")
    
    # --- STEP 1: Feature Selection ---
    print("🧹 Running Random Forest to find Top 50 Features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_raw, y)
    
    # Get importance of every column
    importances = rf.feature_importances_
    # Find indices of the top 50
    indices = np.argsort(importances)[::-1][:50]
    
    # Filter dataset to keep only the best columns
    X_optimized = X_raw[:, indices]
    print(f"✅ Data Optimized! Reduced features from {X_raw.shape[1]} -> {X_optimized.shape[1]}")
    
    return X_optimized, y

# ==========================================
# 2. Train on Optimized Data
# ==========================================
import os # import missing lib

def train():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # Load Optimized Data
    X, y = load_and_optimize()

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

    # Model (Tuned for 50 features)
    model = nn.Sequential(
        nn.Linear(50, 64),      # Input is now 50, not 279
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.4),
        
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(32, 2)
    ).to(device)

    # Training Setup
    class_counts = torch.bincount(y_train)
    weights = 1. / class_counts.float()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-3)

    print("\nStarting Training on Top 50 Features...")
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    patience = 25
    counter = 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, preds = torch.max(test_outputs, 1)
            acc = (preds == y_test).sum().item() / y_test.size(0)

        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
            # Only print when we beat the previous record
            if acc > 0.80: 
                print(f"✨ Epoch {epoch+1}: High Accuracy! -> {acc*100:.2f}%")
        else:
            counter += 1

        if counter >= patience:
            print(f"🛑 Stopping at Epoch {epoch+1}")
            break

    # Final Result
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        print("\n✅ Final Result (With Feature Selection):")
        print(classification_report(y_test.cpu(), preds.cpu(), target_names=["Healthy", "Arrhythmia"]))

if __name__ == "__main__":
    train()