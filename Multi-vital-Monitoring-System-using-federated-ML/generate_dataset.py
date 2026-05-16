import pandas as pd
import numpy as np
import os

def generate_patient_dataset(n_samples=1000, save_path="datasets/patient_data.csv"):
    np.random.seed(42)

    # Features
    spo2 = np.random.normal(loc=97, scale=2, size=n_samples)   # Oxygen saturation
    ecg = np.random.normal(loc=0.5, scale=0.5, size=n_samples) # ECG feature (normalized)
    temp = np.random.normal(loc=37, scale=0.5, size=n_samples) # Body temperature
    pulse = np.random.normal(loc=75, scale=10, size=n_samples) # Heart rate

    # Label rule (just for simulation)
    labels = []
    for s, e, t, p in zip(spo2, ecg, temp, pulse):
        if s < 94 or t > 38.5 or p > 100:
            labels.append(1)  # abnormal
        else:
            labels.append(0)  # normal

    df = pd.DataFrame({
        "spo2": spo2,
        "ecg": ecg,
        "temp": temp,
        "pulse": pulse,
        "label": labels
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Dataset generated and saved at {save_path}")

if __name__ == "__main__":
    generate_patient_dataset(1000)
