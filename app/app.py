# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from inference import InferenceEngine
from config import SETTINGS

app = FastAPI(title="Sensor GMM Baseline API", version="1.0")

engine: Optional[InferenceEngine] = None

class PredictRequest(BaseModel):
    # Generic dict allows "Sensor 0".."Sensor 19"
    features: Dict[str, float] = Field(..., description="Sensor values, e.g., {'Sensor 0': 0.12, ...}")

class PredictResponse(BaseModel):
    cluster: int
    predicted_class: Optional[int]
    confidence: float
    auto_assign: bool
    reason: str
    cluster_support: int
    cluster_purity: Optional[float]

class PredictBatchRequest(BaseModel):
    items: List[PredictRequest]

@app.on_event("startup")
def load_model():
    global engine
    engine = InferenceEngine(model_dir=SETTINGS.OUTDIR)

@app.get("/health")
def health():
    return {"status": "ok", "model_dir": SETTINGS.OUTDIR}

@app.get("/model-info")
def model_info():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "K": engine.K,
        "confidence_threshold": SETTINGS.CONF_THRESH,
        "min_support": SETTINGS.MIN_SUPPORT,
        "min_purity": SETTINGS.MIN_PURITY,
        "self_train": SETTINGS.SELF_TRAIN,
        "pseudo_conf_threshold": SETTINGS.PSEUDO_CONF_THRESH,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        res = engine.predict_one(req.features)  # type: ignore
        return PredictResponse(
            cluster=res.cluster,
            predicted_class=res.predicted_class,
            confidence=res.confidence,
            auto_assign=res.auto_assign,
            reason=res.reason,
            cluster_support=res.cluster_support,
            cluster_purity=res.cluster_purity,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch", response_model=List[PredictResponse])
def predict_batch(req: PredictBatchRequest):
    try:
        rows = [x.features for x in req.items]
        results = engine.predict_batch(rows)  # type: ignore
        return [
        PredictResponse(
            cluster=r.cluster,
            predicted_class=r.predicted_class,
            confidence=r.confidence,
            auto_assign=r.auto_assign,
                reason=r.reason,
                cluster_support=r.cluster_support,
                cluster_purity=r.cluster_purity,
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
