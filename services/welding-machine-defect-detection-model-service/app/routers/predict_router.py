from fastapi import APIRouter
from app.schemas.input import SensorData
from app.services.predict_service import predict_anomaly

router = APIRouter()


@router.post("/predict")
def predict(data: SensorData):
    return predict_anomaly(data.signal_type, data.values)
