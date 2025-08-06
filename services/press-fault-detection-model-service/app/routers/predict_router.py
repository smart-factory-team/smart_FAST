from fastapi import APIRouter, HTTPException

from app.schemas.input import SensorData
from app.schemas.output import PredictionResponse
from app.services.predict_service import predict

router = APIRouter(
    prefix="/predict",
    tags=["prediction"]
)

@router.post("", response_model=PredictionResponse)
async def predict_fault(request_body: SensorData):
    
    try:
        result = predict(data=request_body)
        return PredictionResponse(**result)
    
    except RuntimeError as e:
        # 모델이 로드되지 않았을 경우
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        # 그 외 예측 과정에서 발생할 수 있는 모든 오류
        raise HTTPException(status_code=500, detail=f"예측 처리 중 오류 발생: {str(e)}") from e 