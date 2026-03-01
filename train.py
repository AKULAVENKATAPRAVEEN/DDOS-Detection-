import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils import get_logger, save_artifact, load_artifact, PROC_DIR, MODEL_DIR, encode_labels
from features import get_feature_matrix

logger = get_logger("train")

XGB_PARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, eval_metric="mlogloss",
    early_stopping_rounds=20,
)

LGBM_PARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, random_state=42, verbose=-1,
)

def prepare_data():
    feature_cols = load_artifact("feature_cols.pkl")
    scaler       = load_artifact("scaler.pkl")
    label_map    = load_artifact("label_map.pkl")
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df  = pd.read_csv(PROC_DIR / "test.csv")
    X_train, y_train = get_feature_matrix(train_df)
    X_test,  y_test  = get_feature_matrix(test_df)
    for col in feature_cols:
        if col not in X_train.columns:
            X_train[col] = 0
            X_test[col]  = 0
    X_train = X_train[feature_cols]
    X_test  = X_test[feature_cols]
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    y_train_enc, _ = encode_labels(y_train)
    y_test_enc,  _ = encode_labels(y_test)
    return X_train_s, X_test_s, y_train_enc, y_test_enc, label_map

def evaluate(model, X_test, y_test, label_map, model_name):
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="macro")
    labels  = sorted(label_map.keys())
    names   = [label_map[l] for l in labels]
    report  = classification_report(y_test, y_pred, target_names=names, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    logger.info(f"\n{'='*60}\n  {model_name}\n{'='*60}")
    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  F1-Macro : {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=names)}")
    return {"model": model_name, "accuracy": round(acc,4), "f1_macro": round(f1,4),
            "report": report, "confusion_matrix": cm}

def train():
    logger.info("=== Starting Model Training ===")
    X_train, X_test, y_train, y_test, label_map = prepare_data()
    n_classes = len(label_map) - (1 if -1 in label_map else 0)
    obj = "multi:softmax" if n_classes > 2 else "binary:logistic"
    results = {}
    models  = {}
    split   = int(len(X_train) * 0.9)

    logger.info("Training XGBoost ...")
    xgb = XGBClassifier(objective=obj, num_class=n_classes if n_classes > 2 else None, **XGB_PARAMS)
    xgb.fit(X_train[:split], y_train[:split], eval_set=[(X_train[split:], y_train[split:])], verbose=50)
    results["xgboost"] = evaluate(xgb, X_test, y_test, label_map, "XGBoost")
    models["xgboost"]  = xgb

    logger.info("Training LightGBM ...")
    lgbm = LGBMClassifier(**LGBM_PARAMS)
    lgbm.fit(X_train[:split], y_train[:split], eval_set=[(X_train[split:], y_train[split:])])
    results["lightgbm"] = evaluate(lgbm, X_test, y_test, label_map, "LightGBM")
    models["lightgbm"]  = lgbm

    best_name  = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = models[best_name]
    logger.info(f"\nBest model: {best_name} (F1={results[best_name]['f1_macro']:.4f})")

    save_artifact(best_model,         "best_model.pkl")
    save_artifact(models["xgboost"],  "xgboost_model.pkl")
    save_artifact(models["lightgbm"], "lightgbm_model.pkl")

    eval_path = MODEL_DIR / "eval_report.json"
    with open(eval_path, "w") as f:
        json.dump({"best": best_name, "results": results}, f, indent=2)
    logger.info(f"Saved to {eval_path}")
    logger.info("=== Training Complete ===")
    return best_model, results

if __name__ == "__main__":
    train()
