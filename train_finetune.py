import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os

# ==========================================
# 1. Data Setup: Simulate Two Domains
# ==========================================
def get_cross_domain_data():
    print("🔍 Loading data for Cross-Domain Experiment...")
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
    
    print(f"   Total Data: {X_raw.shape}")

    # --- SIMULATE DOMAINS ---
    # We split data not randomly, but by "Patient ID" chunks to simulate 
    # two different data sources (Source Domain vs Target Domain)
    
    # Source Domain (Hospital A - 70% of data) used for PRE-TRAINING
    split_idx = int(len(X_raw) * 0.7)
    
    X_source = X_raw[:split_idx]
    y_source = y[:split_idx]
    
    # Target Domain (Hospital B - 30% of data) used for FINE-TUNING
    X_target = X_raw[split_idx:]
    y_target = y[split_idx:]
    
    print(f"   🏛️ Source Domain (Pre-Train): {X_source.shape}")
    print(f"   🏥 Target Domain (Fine-Tune): {X_target.shape}")

    # Scale independently (As if they are different hospitals)
    scaler_source = StandardScaler()
    X_source = scaler_source.fit_transform(X_source)
    
    scaler_target = StandardScaler()
    X_target = scaler_target.fit_transform(X_target) # Fit on target separately!

    return X_source, y_source, X_target, y_target

# ==========================================
# 2. Define The Transfer Model
# ==========================================
class TransferModel(nn.Module):
    def __init__(self, input_dim):
        super(TransferModel, self).__init__()
        # Feature Extractor (The "Brain")
        self.features = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Classifier (The "Output")
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out

# ==========================================
# 3. Execution: Pre-Train -> Fine-Tune
# ==========================================
def run_transfer_learning():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    # 1. Get Data
    X_s, y_s, X_t, y_t = get_cross_domain_data()
    
    # Convert to Tensor
    t_Xs = torch.tensor(X_s, dtype=torch.float32).to(device)
    t_ys = torch.tensor(y_s, dtype=torch.long).to(device)
    t_Xt = torch.tensor(X_t, dtype=torch.float32).to(device)
    t_yt = torch.tensor(y_t, dtype=torch.long).to(device)

    # 2. Initialize Model
    model = TransferModel(input_dim=X_s.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # ==========================================
    # PHASE 1: PRE-TRAINING (Source Domain)
    # ==========================================
    print("\n🚀 Phase 1: Pre-Training on Source Domain...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(t_Xs), t_ys)
        loss.backward()
        optimizer.step()
        if (epoch+1)%20==0: print(f"   Epoch {epoch+1}: Loss {loss.item():.4f}")

    print("✅ Pre-Training Complete. Model has learned Source patterns.")

    # ==========================================
    # PHASE 2: FINE-TUNING (Target Domain)
    # ==========================================
    print("\n🔄 Phase 2: Fine-Tuning on Target Domain...")
    
    # FREEZE the Feature Extractor (Keep the learned brain)
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Only train the Classifier (The last layer)
    optimizer_ft = optim.Adam(model.classifier.parameters(), lr=0.005) # Higher LR for head
    
    best_acc = 0.0
    
    for epoch in range(50):
        model.train()
        optimizer_ft.zero_grad()
        loss = criterion(model(t_Xt), t_yt)
        loss.backward()
        optimizer_ft.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            preds = model(t_Xt).argmax(dim=1)
            acc = (preds == t_yt).sum().item() / len(t_yt)
            if acc > best_acc: best_acc = acc
            
    print(f"✨ Fine-Tuning Complete. Best Target Accuracy: {best_acc*100:.2f}%")

    # ==========================================
    # 4. Final Evaluation & Plots
    # ==========================================
    model.eval()
    with torch.no_grad():
        logits = model(t_Xt)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = logits.argmax(dim=1).numpy()
        
    print("\n✅ Final Cross-Domain Report:")
    print(classification_report(y_t, preds, target_names=["Healthy", "Arrhythmia"]))

    # PLOT 1: Confusion Matrix
    cm = confusion_matrix(y_t, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
    plt.title('Cross-Domain Fine-Tuning CM')
    plt.ylabel('Actual (Target Domain)')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('finetune_cm.png', dpi=300)
    print("📸 Saved 'finetune_cm.png'")

    # PLOT 2: ROC
    fpr, tpr, _ = roc_curve(y_t, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'Fine-Tuned ROC (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('Cross-Domain Fine-Tuning ROC')
    plt.legend()
    plt.savefig('finetune_roc.png', dpi=300)
    print("📸 Saved 'finetune_roc.png'")

if __name__ == "__main__":
    run_transfer_learning()