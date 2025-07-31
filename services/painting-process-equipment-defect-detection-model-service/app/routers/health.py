from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_model, get_config

router = APIRouter()

@router.get("/health")
async def health_check(
    model_status: None = Depends(get_model),
    config_status: None = Depends(get_config)
):
    """
    서비스의 헬스 상태를 확인합니다.
    모델과 설정이 정상적으로 로드되었는지도 함께 확인합니다.
    """
    return {"status": "ok", "message": "Service is healthy."}