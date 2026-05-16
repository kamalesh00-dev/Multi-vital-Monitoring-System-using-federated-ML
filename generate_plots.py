import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ==========================================
# 1. Load & Optimize (Same as before)
# ==========================================
def load_and_optimize():
    print("🔍 Loading data...")
    path = r".\datasets\health_status\arrhythmia\arrhythmia_clean.csv"
    
    # Auto-find file
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
    
    # Feature Selection (Top 50)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_raw, y)
    indices = np.argsort(rf.feature_importances_)[::-1][:50]
    X_optimized = X_raw[:, indices]
    
    return X_optimized, y

# ==========================================
# 2. Train & Plot
# ==========================================
import os

def generate_visuals():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu") # CPU is fine for plotting

    # Load
    X, y = load_and_optimize()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Define Model (The successful 86% architecture)
    model = nn.Sequential(
        nn.Linear(50, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 2)
    ).to(device)

    # Train (Fast version)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0])) # Simplified weights

    print("⚡ Retraining model to generate plots...")
    for epoch in range(60): # 60 epochs was your sweet spot
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Predict probabilities for ROC
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy() # Probability of 'Arrhythmia'
        preds = logits.argmax(dim=1).numpy()

    # --- PLOT 1: CONFUSION MATRIX ---
    print("🎨 Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Arrhythmia'], yticklabels=['Healthy', 'Arrhythmia'])
    plt.title('Confusion Matrix (Top 50 Features)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ Saved 'confusion_matrix.png'")

    # --- PLOT 2: ROC CURVE ---
    print("🎨 Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    print("✅ Saved 'roc_curve.png'")
    
    print("\n🎉 DONE! You can now put these images in your paper.")

if __name__ == "__main__":
    generate_visuals()