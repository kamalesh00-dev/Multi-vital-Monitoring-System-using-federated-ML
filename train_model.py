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
# 1. Smart Searcher (Finds the numeric file)
# ==========================================
def find_valid_data():
    print("🔍 Scanning project for valid data files...")
    
    # Walk through all folders in the project
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                
                # Check 1: Is it the right size? (Scripts are small, Data is big)
                if os.path.getsize(full_path) < 2000: # Skip files smaller than 2KB (likely scripts)
                    continue
                
                # Check 2: Peek inside for numbers
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline().strip()
                        # If first line contains "import" or "#", it's code. Skip it.
                        if "import " in first_line or "#" in first_line:
                            continue
                        # If it has many commas, it's likely data
                        if first_line.count(',') > 10:
                            print(f"✅ FOUND REAL DATA: {full_path}")
                            print(f"   (Preview: {first_line[:50]}...)")
                            return full_path
                except:
                    continue

    print("❌ CRITICAL: Could not find any file with real data.")
    print("   Please ensure 'arrhythmia_clean.csv' is saved and contains numbers.")
    sys.exit(1)

# ==========================================
# 2. Load & Train
# ==========================================
def train():
    # 1. Find the file automatically
    data_path = find_valid_data()
    
    print(f"🚀 Loading data from: {data_path}")
    
    # 2. Load with Pandas
    try:
        # engine='python' handles uneven rows better
        df = pd.read_csv(data_path, engine='python')
        
        # CLEANING: Drop errors/text rows
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        print(f"   Shape: {df.shape}")
        
        if len(df) < 10:
            print("❌ File was found but seems empty of valid numbers.")
            sys.exit(1)
            
        # 3. Prepare Tensors
        X = df.iloc[:, :-1].values # Features
        y = df.iloc[:, -1].values  # Target (Last col)
        
        # Encode labels (0, 1, 2...)
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        # 4. Define Model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y))
        print(f"ℹ️ Model Config: Features={input_dim}, Classes={num_classes}")
        
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        ).to(device)

        # 5. Weighted Loss
        class_counts = torch.bincount(y_train)
        weights = 1. / class_counts.float()
        weights = weights / weights.sum() * len(class_counts)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        # 6. Train
        print("\nStarting Training...")
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

        # 7. Report
        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(dim=1)
            print("\n🔍 Results on Test Data:")
            print(classification_report(y_test.cpu(), preds.cpu(), zero_division=0))

    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    train()