from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from .inference import ModelRunner

MODEL_PATH = "./app/artifacts/model_bundle.joblib"

app = FastAPI()
runner = ModelRunner.load(MODEL_PATH)


class PredictRequest(BaseModel):
    # Send either single or batch payload
    rows: List[Dict[str, float]]


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    preds = runner.predict(req.rows)
    return {
        "predictions": [p.__dict__ for p in preds]
    }

