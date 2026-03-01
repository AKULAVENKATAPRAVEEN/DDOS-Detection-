import os
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
MODEL_DIR  = ROOT_DIR / "models"

for d in [RAW_DIR, PROC_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BENIGN_LABEL = "BENIGN"

ATTACK_TYPES = [
    "DrDoS_DNS", "DrDoS_LDAP", "DrDoS_MSSQL", "DrDoS_NetBIOS",
    "DrDoS_NTP", "DrDoS_SNMP", "DrDoS_SSDP", "DrDoS_UDP",
    "Syn", "TFTP", "UDP-lag", "WebDDoS",
]

DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP", "Source Port",
    "Destination Port", "Timestamp", "SimillarHTTP", "Unnamed: 0",
]

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)

def save_artifact(obj, filename: str):
    path = MODEL_DIR / filename
    joblib.dump(obj, path)
    return path

def load_artifact(filename: str):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)

def load_raw_csvs(directory: Path = RAW_DIR) -> pd.DataFrame:
    files = list(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    logger = get_logger("utils")
    dfs = []
    for f in files:
        logger.info(f"Loading {f.name} ...")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined):,}")
    return combined

def encode_labels(series: pd.Series, binary: bool = False):
    if binary:
        encoded = (series != BENIGN_LABEL).astype(int).values
        label_map = {0: BENIGN_LABEL, 1: "ATTACK"}
    else:
        classes = [BENIGN_LABEL] + sorted(series[series != BENIGN_LABEL].unique().tolist())
        label_map = {i: c for i, c in enumerate(classes)}
        inv_map   = {c: i for i, c in label_map.items()}
        encoded   = series.map(inv_map).fillna(-1).astype(int).values
        label_map[-1] = "UNKNOWN"
    return encoded, label_map
