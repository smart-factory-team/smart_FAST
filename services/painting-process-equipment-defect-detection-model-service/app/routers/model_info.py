from fastapi import APIRouter, Depends
from ..dependencies import get_config

router = APIRouter()

@router.get("/")
async def get_model_info(
    config: dict = Depends(get_config) # 설정 의존성 주입
):
    """
    로드된 AI 모델의 설정 정보를 반환합니다.
    """
    # model_config.yaml의 내용을 직접 반환 (일반적으로 필요한 정보만 필터링해서 반환)
    # 여기서는 전체 config에서 model, training, threshold, features 정보를 반환
    model_details = {
        "model": config.get("model"),
        "training": config.get("training"),
        "threshold": config.get("threshold"),
        "features": config.get("features")
    }
    return model_details