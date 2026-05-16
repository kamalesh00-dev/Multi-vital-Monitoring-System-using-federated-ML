import pandas as pd
import os

# --- File paths ---
data_path = os.path.join("datasets", "health_status", "arrhythmia", "arrhythmia.data")
save_path = os.path.join("datasets", "health_status", "arrhythmia", "arrhythmia_clean.csv")

# --- Step 1: Load the raw UCI arrhythmia dataset ---
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found: {data_path}")

# The .data file is comma-separated and may have '?'
df = pd.read_csv(data_path, header=None, na_values="?")

print(f"Raw shape: {df.shape}")

# --- Step 2: Clean up missing and invalid data ---
df = df.dropna(axis=1, how='all')  # remove empty columns
df = df.dropna(thresh=int(0.5 * df.shape[1]))  # remove rows with too many missing
df = df.fillna(df.mean(numeric_only=True))  # fill missing with mean

# --- Step 3: Rename last column to 'Status' (disease category) ---
df.rename(columns={df.columns[-1]: "Status"}, inplace=True)

# --- Step 4: Save the cleaned CSV ---
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)

print(f"✅ Cleaned dataset saved to: {save_path}")
print(f"Cleaned shape: {df.shape}")
