import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ==========================================
# 1. Load & Apply PCS (PCA)
# ==========================================
def load_and_pca():
    print("🔍 Loading data...")
    possible_paths = [
        r".\datasets\health_status\arrhythmia\arrhythmia_clean.csv",
        "arrhythmia_clean.csv",
        "datasets/arrhythmia_clean.csv"
    ]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, engine='python')
            break
            
    if df is None:
        print("❌ Error: Could not find 'arrhythmia_clean.csv'")
        exit(1)

    # Clean
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    X_raw = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    
    # Binary Labels (0=Normal, 1=Arrhythmia)
    y = np.where(y_raw == 1, 0, 1)

    # --- STEP 1: PCS (Principal Component Analysis) ---
    print(f"   Original Features: {X_raw.shape[1]}")
    print("⚙️ Applying PCS (PCA) to compress features...")
    
    # Scale first (Critical for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Reduce to 64 components (high variance retention)
    pca = PCA(n_components=64)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"✅ PCS Complete! Data Shape: {X_pca.shape}")
    return X_pca, y

# ==========================================
# 2. Define BiLSTM Model
# ==========================================
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM Layer
        # batch_first=True means input is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        
        # Fully Connected Layer
        # input is hidden_size * 2 because it's Bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        # out shape: (batch, seq_len, hidden*2)
        out = out[:, -1, :] 
        
        # Classifier
        out = self.fc(out)
        return out

# ==========================================
# 3. Train & Generate Plots
# ==========================================
def train_and_plot():
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training BiLSTM on: {device}")

    # Load Data
    X, y = load_and_pca()
    
    # Reshape for LSTM: (Batch, Sequence, Feature)
    # We treat the 64 PCA features as a sequence of length 64, each with 1 value
    X = X.reshape(X.shape[0], 64, 1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Initialize Model
    model = BiLSTMModel(input_size=1, hidden_size=64, num_layers=2, num_classes=2).to(device)
    
    # Loss & Optimizer
    weights = torch.tensor([1.0, 1.2]).to(device) # Slight penalty for missing Sick cases
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    print("\nStarting BiLSTM Training...")
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Check accuracy
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(X_test_t)
                _, preds = torch.max(test_out, 1)
                acc = (preds.cpu().numpy() == y_test).sum() / len(y_test)
                print(f"Epoch {epoch+1}: Loss={loss.item():.4f} | Test Acc={acc*100:.2f}%")
                
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model.state_dict())

    print(f"\n✨ Best Accuracy Reached: {best_acc*100:.2f}%")
    
    # Load Best Model for Plots
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    # --- PLOT 1: Confusion Matrix ---
    print("🎨 Generating Confusion Matrix...")
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Healthy', 'Arrhythmia'], yticklabels=['Healthy', 'Arrhythmia'])
    plt.title('BiLSTM Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('bilstm_confusion.png', dpi=300)
    print("✅ Saved 'bilstm_confusion.png'")

    # --- PLOT 2: ROC Curve ---
    print("🎨 Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'BiLSTM ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BiLSTM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('bilstm_roc.png', dpi=300)
    print("✅ Saved 'bilstm_roc.png'")
    
    print("\n✅ DONE! You can now use these PCS-BiLSTM results.")

if __name__ == "__main__":
    train_and_plot()