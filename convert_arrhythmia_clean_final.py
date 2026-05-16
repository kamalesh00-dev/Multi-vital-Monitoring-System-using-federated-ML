import pandas as pd
import numpy as np
import os

# Path setup
src_path = r"datasets/health_status/arrhythmia/arrhythmia.data"
save_path = r"datasets/health_status/arrhythmia/arrhythmia_clean.csv"

# Read file manually since it's not a standard CSV
rows = []
with open(src_path, "r") as f:
    for line in f:
        parts = [x.strip() for x in line.strip().split(",")]
        if len(parts) > 1:  # ignore empty or single-value lines
            rows.append(parts)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Drop columns that are entirely empty or have single unique value
df = df.replace("?", np.nan)
df = df.dropna(axis=1, how='all')

# Try to convert all columns to numeric where possible
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows with too many NaN (more than 40%)
df = df.dropna(thresh=int(df.shape[1] * 0.6))

# Fill remaining NaNs with mean
df = df.fillna(df.mean())

# For model training, we need a target label — assume last column is target
if df.shape[1] > 1:
    df.columns = [f"f{i}" for i in range(df.shape[1]-1)] + ["Status"]

# Convert target to integer classes
df["Status"] = pd.factorize(df["Status"])[0]

print(f"✅ Cleaned Arrhythmia dataset shape: {df.shape}")
print(df.head())

# Save cleaned version
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)
print(f"💾 Saved cleaned dataset to: {save_path}")
