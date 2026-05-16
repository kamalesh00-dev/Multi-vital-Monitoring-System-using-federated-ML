import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# ======================================================
# CONFIG
# ======================================================
DATA_DIR = os.path.join("datasets", "health_status", "uci_heart")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

SITE_FILES = {
    "cleveland": ["cleveland.data", "processed.cleveland.data"],
    "hungarian": ["hungarian.data", "processed.hungarian.data"],
    "switzerland": ["switzerland.data", "processed.switzerland.data"],
    "va": ["processed.va.data"]
}


# ======================================================
# HELPERS
# ======================================================
def _load_one(path: str) -> pd.DataFrame:
    """Load a single .data file safely with flexible encoding."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(
            path,
            header=None,
            names=COLUMNS,
            na_values=["?"],
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            header=None,
            names=COLUMNS,
            na_values=["?"],
            encoding="latin1",
            encoding_errors="ignore"
        )
    return df


def load_all_sites() -> Dict[str, pd.DataFrame]:
    """Load and combine datasets for all sites with console logging."""
    print("\n🔍 Loading hospital site datasets...\n")
    sites: Dict[str, pd.DataFrame] = {}

    for site, files in SITE_FILES.items():
        dfs = []
        for fname in files:
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.exists(fpath):
                df = _load_one(fpath)
                if not df.empty:
                    dfs.append(df)
        if len(dfs) == 0:
            print(f"⚠️  Skipped {site.title()} (no valid data files found)")
            continue

        df_site = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {site.title():<12} ({len(df_site)} records)")
        sites[site] = df_site

    if len(sites) == 0:
        print("❌ No hospital datasets found. Check your folder paths.")
    else:
        print("\n✅ All available hospitals loaded successfully!\n")

    return sites


def clean_site(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and binarize site dataset."""
    if df.empty:
        return df

    # Cast numerics
    for col in COLUMNS:
        if col != "num":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing values
    df = df.dropna().copy()

    # Binarize target
    df["num"] = (df["num"].astype(float) > 0).astype(int)
    return df


def train_test_split_per_site(
    df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split one site's data into train/test."""
    if df.empty:
        return df, df
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = max(1, int(len(df) * test_frac))
    test = df.iloc[:n_test]
    train = df.iloc[n_test:]
    return train, test


def make_clients(
    test_frac: float = 0.2, seed: int = 42
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Prepare train splits per site + combined global test set."""
    raw_sites = load_all_sites()
    clients: Dict[str, pd.DataFrame] = {}
    test_parts: List[pd.DataFrame] = []

    for site, df in raw_sites.items():
        df = clean_site(df)
        if df.empty:
            print(f"⚠️  {site.title()} cleaned dataset is empty — skipping client.")
            continue
        train_df, test_df = train_test_split_per_site(df, test_frac=test_frac, seed=seed)
        clients[site] = train_df
        test_parts.append(test_df)

    if len(test_parts) == 0:
        global_test = pd.DataFrame(columns=COLUMNS)
    else:
        global_test = pd.concat(test_parts, ignore_index=True)

    print(f"\n🏥 Total active hospitals: {len(clients)}")
    print(f"🧪 Global test set size: {len(global_test)}\n")

    return clients, global_test


def xy_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convert dataframe to X, y tensors."""
    if df.empty:
        return np.zeros((0, 13), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = df.drop(columns=["num"]).values.astype(np.float32)
    y = df["num"].values.astype(np.int64)
    return X, y
