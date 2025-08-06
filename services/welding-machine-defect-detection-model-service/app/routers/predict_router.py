from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from app.schemas.input import SensorData
from app.services.predict_service import predict_anomaly

router = APIRouter()


class PredictionResponse(BaseModel):
    signal_type: str
    mae: float
    threshold: float
    status: Literal["anomaly", "normal"]


@router.post("/predict", response_model=PredictionResponse)
def predict(data: SensorData):
    try:
        result = predict_anomaly(data.signal_type, data.values)
        return PredictionResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, detail=f"Model files not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")
