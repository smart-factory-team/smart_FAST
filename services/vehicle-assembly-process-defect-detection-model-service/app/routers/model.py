from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import logging

from app.core.config import settings
from app.core.dependencies import get_request_id, get_model_manager
from app.schemas import (
     ModelInfoResponse,
     ModelClassesResponse,
     ModelPerformanceResponse,
     ModelReloadResponse
 )
from app.services.model_service import ModelService

router = APIRouter(prefix="/model", tags=["Model"])
logger = logging.getLogger(__name__)


def get_model_service(
    model_manager = Depends(get_model_manager)
) -> ModelService:
    """모델 서비스 의존성"""
    return ModelService(model_manager)


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info(
    request_id: str = Depends(get_request_id),
    model_service: ModelService = Depends(get_model_service)
):
    """
    모델 정보 조회

    현재 로드된 AI 모델의 상세 정보를 제공합니다.

    **제공 정보:**
    - 모델 이름 및 버전
    - 지원하는 불량 클래스
    - 모델 성능 지표
    - 로딩 시간 및 사용 통계
    - 입력 요구사항
    """

    try:
        logger.info(f"모델 정보 조회 요청 - Request ID: {request_id}")

        # 모델 정보 수집
        model_info = await model_service.get_model_info()

        logger.info(f"모델 정보 조회 완료 - Request ID: {request_id}")

        return ModelInfoResponse(
            success=True,
            message="모델 정보 조회 성공",
            data=model_info,
            timestamp=datetime.now(),
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"모델 정보 조회 중 오류 발생 - Request ID: {request_id}, 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "모델 정보 조회 중 오류가 발생했습니다",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/classes")
async def get_model_classes(
    request_id: str = Depends(get_request_id),
    model_service: ModelService = Depends(get_model_service)
):
    """
    모델이 지원하는 불량 클래스 목록 조회

    **응답:**
    - 클래스 ID와 이름 매핑
    - 각 클래스별 설명
    - 클래스 계층 구조 (있는 경우)
    """

    try:
        logger.info(f"모델 클래스 정보 조회 요청 - Request ID: {request_id}")

        classes_info = await model_service.get_model_classes()

        return ModelClassesResponse(
            success=True,
            message="모델 클래스 정보 조회 성공",
            data=classes_info,  # ModelClassesInfo 객체
            timestamp=datetime.now(),
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"모델 클래스 조회 중 오류 발생 - Request ID: {request_id}, 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "모델 클래스 조회 중 오류가 발생했습니다",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/performance")
async def get_model_performance(
    request_id: str = Depends(get_request_id),
    model_service: ModelService = Depends(get_model_service)
):
    """
    모델 성능 지표 조회

    **제공 지표:**
    - 정확도, 정밀도, 재현율
    - 혼동 행렬
    - 클래스별 성능
    - 처리 속도 통계
    """

    try:
        logger.info(f"모델 성능 조회 요청 - Request ID: {request_id}")

        performance_info = await model_service.get_model_performance()

        return ModelPerformanceResponse(  # ✅ 스키마 객체로 반환
           success=True,
           message="모델 성능 정보 조회 성공",
           data=performance_info,
           timestamp=datetime.now(),
           request_id=request_id
       )

    except Exception as e:
        logger.error(f"모델 성능 조회 중 오류 발생 - Request ID: {request_id}, 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "모델 성능 조회 중 오류가 발생했습니다",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/reload")
async def reload_model(
    request_id: str = Depends(get_request_id),
    model_service: ModelService = Depends(get_model_service)
):
    """
    모델 재로딩 (관리자 기능)

    현재 로드된 모델을 다시 로딩합니다.
    모델 업데이트나 문제 해결 시 사용됩니다.

    **주의사항:**
    - 재로딩 중에는 예측 서비스가 일시적으로 중단됩니다
    - 관리자 권한이 필요합니다
    """

    try:
        logger.info(f"모델 재로딩 요청 - Request ID: {request_id}")

        # 모델 재로딩 수행
        reload_result = await model_service.reload_model()

        logger.info(f"모델 재로딩 완료 - Request ID: {request_id}")

        return ModelReloadResponse(
            success = True,
            message = "모델 재로딩이 완료되었습니다",
            data = reload_result,
            timestamp = datetime.now(),
            request_id = request_id
        )

    except Exception as e:
        logger.error(f"모델 재로딩 중 오류 발생 - Request ID: {request_id}, 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "모델 재로딩 중 오류가 발생했습니다",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        )