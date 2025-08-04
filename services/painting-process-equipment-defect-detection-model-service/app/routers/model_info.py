from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, ValidationError
from app.dependencies import get_config

router = APIRouter()

class ModelInfo(BaseModel):
    model: dict | None = Field(default=None)
    training: dict | None = Field(default=None)
    threshold: dict | None = Field(default=None)
    features: dict | None = Field(default=None)

@router.get("", response_model=ModelInfo, status_code=status.HTTP_200_OK)
def get_model_info(config: dict = Depends(get_config)):
    """
    모델의 설정 정보(모델, 학습, 임계값, 피처 정보)를 반환합니다.
    """
    try:
        filtered_config = {
            "model": config.get("model"),
            "training": config.get("training"),
            "threshold": config.get("threshold"),
            "features": config.get("features"),
        }
        return ModelInfo(**filtered_config)
    except ValidationError as e:
        # Pydantic 유효성 검사 실패 시에만 500 에러 발생
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"모델 설정 유효성 검사 실패: {e}"
        )