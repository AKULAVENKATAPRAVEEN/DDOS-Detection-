import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from predict import get_predictor

app = FastAPI(
    title="DDoS Detection API",
    description="Real-time DDoS detection using XGBoost/LightGBM on CIC-DDoS2019",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class FlowFeatures(BaseModel):
    features: dict[str, float]

class BatchFlowFeatures(BaseModel):
    flows: list[dict[str, float]]

@app.on_event("startup")
async def startup():
    try:
        get_predictor()
    except Exception as e:
        print(f"[WARN] Could not preload model: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model/info")
def model_info():
    try:
        return get_predictor().get_model_info()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict")
def predict_single(body: FlowFeatures):
    try:
        return get_predictor().predict(body.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(body: BatchFlowFeatures):
    if not body.flows:
        raise HTTPException(status_code=400, detail="No flows provided")
    try:
        predictor   = get_predictor()
        predictions = predictor.predict_batch(body.flows)
        attack_count = sum(1 for p in predictions if p["is_attack"])
        return {"predictions": predictions, "total_flows": len(predictions),
                "attack_count": attack_count, "benign_count": len(predictions) - attack_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    try:
        contents    = await file.read()
        df          = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        predictor   = get_predictor()
        predictions = predictor.predict_batch(df)
        attack_count = sum(1 for p in predictions if p["is_attack"])
        return {"predictions": predictions, "total_flows": len(predictions),
                "attack_count": attack_count, "benign_count": len(predictions) - attack_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```



