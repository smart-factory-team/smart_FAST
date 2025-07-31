from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def read_root():
    """
    서비스의 기본 정보 (이름, 버전, 설명)를 반환합니다.
    """
    return {
        "service_name": "painting-process-equipment-defect-detection-model-service",
        "version": "1.0.0",
        "description": "API for analyzing e-coating process parameters, predicting issues, and providing model insights."
    }