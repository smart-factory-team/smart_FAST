from fastapi import APIRouter, Depends
from datetime import datetime

from app.core.config import settings
from app.core.dependencies import get_request_id
from app.schemas import ServiceInfoResponse

router = APIRouter(tags=["Root"])


@router.get("/", response_model=ServiceInfoResponse)
async def get_service_info(request_id: str = Depends(get_request_id)):
    """
    서비스 기본 정보 제공

    - **서비스명**: 자동차 의장 공정 불량 탐지 API
    - **버전**: 현재 서비스 버전
    - **상태**: 서비스 실행 상태
    - **문서**: API 문서 링크
    """

    return ServiceInfoResponse(
        success=True,
        message="자동차 의장 공정 불량 탐지 API 서비스",
        data={
            "service_name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "description": settings.PROJECT_DESCRIPTION,
            "environment": settings.ENVIRONMENT,
            "status": "running",
            "api_docs": {
                "swagger_ui": "/docs" if settings.DEBUG else None,
                "redoc": "/redoc" if settings.DEBUG else None,
                "openapi_json": f"{settings.API_V1_STR}/openapi.json" if settings.DEBUG else None
            },
            "endpoints": {
                "health_check": "/health",
                "readiness_check": "/ready",
                "startup_check": "/startup",
                "prediction": "/predict",
                "file_prediction": "/predict/file",
                "model_info": "/model/info"
            },
            "features": [
                "이미지 기반 불량 탐지",
                "실시간 예측",
                "배치 처리 지원",
                "HuggingFace 모델 지원"
            ]
        },
        timestamp=datetime.now(),
        request_id=request_id
    )