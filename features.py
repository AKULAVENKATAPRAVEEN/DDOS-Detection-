import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from utils import get_logger, save_artifact, load_artifact, PROC_DIR, encode_labels

logger = get_logger("features")

PRIORITY_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

def engineer_features(df):
    eps = 1e-9
    if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
        total = df["Total Fwd Packets"] + df["Total Backward Packets"] + eps
        df["Fwd_Pkt_Ratio"] = df["Total Fwd Packets"] / total
        df["Bwd_Pkt_Ratio"] = df["Total Backward Packets"] / total
    if "Total Length of Fwd Packets" in df.columns and "Total Length of Bwd Packets" in df.columns:
        total_bytes = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"] + eps
        df["Fwd_Byte_Ratio"] = df["Total Length of Fwd Packets"] / total_bytes
    if "Fwd Packets/s" in df.columns and "Bwd Packets/s" in df.columns:
        df["Pkt_Rate_Ratio"] = df["Fwd Packets/s"] / (df["Bwd Packets/s"] + eps)
    flag_cols = [c for c in df.columns if "Flag" in c]
    if flag_cols:
        df["Total_Flags"] = df[flag_cols].sum(axis=1)
    return df

def get_feature_matrix(df):
    y = df["Label"]
    X = df.drop(columns=["Label"])
    X = engineer_features(X)
    X = X.select_dtypes(include=[np.number])
    available = X.columns.tolist()
    extra_eng = [c for c in available if c not in PRIORITY_FEATURES]
    selected = [c for c in PRIORITY_FEATURES if c in available] + extra_eng
    X = X[selected]
    return X, y

def build_and_save(use_feature_selection=True):
    logger.info("=== Starting Feature Engineering ===")
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    X_train, y_train = get_feature_matrix(train_df)
    y_enc, label_map = encode_labels(y_train, binary=False)
    logger.info(f"Raw feature count: {X_train.shape[1]}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    feature_cols = X_train.columns.tolist()
    if use_feature_selection:
        logger.info("Running XGBoost feature importance selection ...")
        selector_model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, eval_metric="mlogloss",
            use_label_encoder=False
        )
        selector_model.fit(X_scaled, y_enc)
        selector = SelectFromModel(selector_model, prefit=True, threshold="mean")
        mask = selector.get_support()
        feature_cols = [f for f, m in zip(feature_cols, mask) if m]
        logger.info(f"Selected {len(feature_cols)} features")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(feature_cols, "feature_cols.pkl")
    save_artifact(label_map, "label_map.pkl")
    logger.info("=== Feature Engineering Complete ===")
    return feature_cols, scaler, label_map

if __name__ == "__main__":
    build_and_save()
