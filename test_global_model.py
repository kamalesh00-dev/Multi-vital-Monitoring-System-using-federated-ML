# test_global_model.py
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import PatientModel  # import your model class

# Metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def main():
    model_path = "global_model.pth"

    # Check model file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train_model.py first.")

    # 1) Load model
    model = PatientModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2) Load dataset
    df = pd.read_csv("datasets/patient_data.csv")
    X = df[["spo2", "ecg", "temp", "pulse"]].values
    y = df["label"].values

    # Train-test split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to torch tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 3) Inference
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1).numpy()

    # 4) Metrics
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, digits=4)

    # Print summary
    print("\n=== Inference Summary ===")
    print(f"Number of test samples: {len(y_test)}")
    print(f"Accuracy: {acc:.4f}\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # 5) Save predictions
    out_df = pd.DataFrame({
        "pred": preds,
        "true": y_test,
        "spo2": X_test[:,0],
        "ecg": X_test[:,1],
        "temp": X_test[:,2],
        "pulse": X_test[:,3],
    })
    out_csv = "saved_models/predictions.csv"
    os.makedirs("saved_models", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved predictions to {out_csv}")

    # 6) Confusion matrix heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Abnormal"],
                yticklabels=["Normal","Abnormal"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # 7) Feature distributions
    features = ["spo2", "ecg", "temp", "pulse"]
    for feat in features:
        plt.figure(figsize=(7,4))
        sns.histplot(data=out_df, x=feat, hue="true", bins=20, kde=True, element="step")
        plt.title(f"{feat.upper()} distribution by True Label")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
