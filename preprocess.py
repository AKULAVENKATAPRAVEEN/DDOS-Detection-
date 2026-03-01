import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_logger, load_raw_csvs, save_artifact, DROP_COLS, PROC_DIR, BENIGN_LABEL

logger = get_logger("preprocess")

def normalize_columns(df):
    df.columns = df.columns.str.strip()
    return df

def drop_irrelevant(df):
    to_drop = [c for c in DROP_COLS if c in df.columns]
    logger.info(f"Dropping columns: {to_drop}")
    return df.drop(columns=to_drop)

def fix_label(df):
    label_col = None
    for candidate in ["Label", "label", " Label"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("No 'Label' column found in dataset.")
    df = df.rename(columns={label_col: "Label"})
    df["Label"] = df["Label"].str.strip()
    return df

def clean_values(df):
    label = df["Label"]
    features = df.drop(columns=["Label"])
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(features)
    features.dropna(inplace=True)
    after = len(features)
    logger.info(f"Dropped {before - after:,} rows with NaN/Inf values")
    df = features.join(label.loc[features.index])
    return df

def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df):,} duplicate rows")
    return df

def balance_classes(df, max_benign_ratio=2.0):
    attack_count = (df["Label"] != BENIGN_LABEL).sum()
    max_benign = int(attack_count * max_benign_ratio)
    benign_df = df[df["Label"] == BENIGN_LABEL]
    if len(benign_df) > max_benign:
        benign_df = benign_df.sample(n=max_benign, random_state=42)
        df = pd.concat([benign_df, df[df["Label"] != BENIGN_LABEL]], ignore_index=True)
        logger.info(f"Downsampled BENIGN to {max_benign:,} rows")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def run_pipeline(balance=True):
    logger.info("=== Starting Preprocessing Pipeline ===")
    df = load_raw_csvs()
    df = normalize_columns(df)
    df = fix_label(df)
    df = drop_irrelevant(df)
    df = clean_values(df)
    df = remove_duplicates(df)
    if balance:
        df = balance_classes(df)
    dist = df["Label"].value_counts()
    logger.info(f"Class distribution:\n{dist.to_string()}")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Label"])
    train_df.to_csv(PROC_DIR / "train.csv", index=False)
    test_df.to_csv(PROC_DIR / "test.csv", index=False)
    logger.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    logger.info("=== Preprocessing Complete ===")
    return df

if __name__ == "__main__":
    run_pipeline()
