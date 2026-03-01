import time
import numpy as np
import pandas as pd
from utils import get_logger, load_artifact
from features import engineer_features

logger = get_logger("predict")

class DDoSPredictor:
    def __init__(self):
        logger.info("Loading model artifacts ...")
        self.model         = load_artifact("best_model.pkl")
        self.scaler        = load_artifact("scaler.pkl")
        self.feature_cols  = load_artifact("feature_cols.pkl")
        self.label_map     = load_artifact("label_map.pkl")
        logger.info(f"Predictor ready | {len(self.feature_cols)} features")

    def _preprocess(self, df):
        df = df.select_dtypes(include=[np.number])
        df = engineer_features(df)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_cols]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return self.scaler.transform(df)

    def predict(self, flow):
        t0 = time.perf_counter()
        if isinstance(flow, dict):
            flow = pd.DataFrame([flow])
        X = self._preprocess(flow)
        if hasattr(self.model, "predict_proba"):
            probs_arr  = self.model.predict_proba(X)[0]
            pred_class = int(np.argmax(probs_arr))
            confidence = float(np.max(probs_arr))
            probs_dict = {self.label_map.get(i, str(i)): round(float(p), 4)
                          for i, p in enumerate(probs_arr)}
        else:
            pred_class = int(self.model.predict(X)[0])
            confidence = 1.0
            probs_dict = {}
        label     = self.label_map.get(pred_class, "UNKNOWN")
        is_attack = label != "BENIGN"
        latency   = (time.perf_counter() - t0) * 1000
        return {"label": label, "is_attack": is_attack,
                "confidence": round(confidence, 4),
                "probabilities": probs_dict, "latency_ms": round(latency, 3)}

    def predict_batch(self, flows):
        if isinstance(flows, list):
            df = pd.DataFrame(flows)
        else:
            df = flows.copy()
        X = self._preprocess(df)
        if hasattr(self.model, "predict_proba"):
            probs_arr    = self.model.predict_proba(X)
            pred_classes = np.argmax(probs_arr, axis=1)
        else:
            pred_classes = self.model.predict(X)
            probs_arr    = None
        results = []
        for i, pred in enumerate(pred_classes):
            label     = self.label_map.get(int(pred), "UNKNOWN")
            is_attack = label != "BENIGN"
            confidence = float(np.max(probs_arr[i])) if probs_arr is not None else 1.0
            probs_dict = ({self.label_map.get(j, str(j)): round(float(p), 4)
                           for j, p in enumerate(probs_arr[i])}
                          if probs_arr is not None else {})
            results.append({"label": label, "is_attack": is_attack,
                             "confidence": round(confidence, 4), "probabilities": probs_dict})
        return results

    def get_model_info(self):
        return {"model_type": type(self.model).__name__,
                "feature_count": len(self.feature_cols),
                "classes": list(self.label_map.values()),
                "features": self.feature_cols}

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = DDoSPredictor()
    return _predictor
