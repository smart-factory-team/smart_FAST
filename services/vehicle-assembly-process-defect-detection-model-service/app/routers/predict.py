from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from datetime import datetime
from typing import List
import logging

from app.core.config import settings
from app.core.dependencies import get_request_id, get_model_manager
from app.schemas import (
    PredictRequest,
    PredictResponse,
    FilePredictResponse,
    BatchPredictResponse
)
from app.services.prediction_service import PredictionService
from app.core.exceptions import (
    InvalidImageException,
    ImageSizeException,
    UnsupportedImageFormatException
)

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)


def get_prediction_service(
    model_manager = Depends(get_model_manager)
) -> PredictionService:
    """예측 서비스 의존성"""
    return PredictionService(model_manager)


@router.post("", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    request_id: str = Depends(get_request_id),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    AI 모델 예측 (JSON 데이터)

    이미지 URL을 받아서 불량을 탐지합니다.

    **지원하는 입력:**
    - 이미지 URL (공개 접근 가능한 URL)

    **응답:**
    - 예측된 불량 클래스
    - 신뢰도 점수
    - 각 클래스별 확률
    """

    try:
        logger.info(f"예측 요청 시작 - Request ID: {request_id}")

        # 예측 수행
        result = await prediction_service.predict_from_data(request)

        logger.info(
            f"예측 완료 - Request ID: {request_id}, "
            f"결과: {result.data.predicted_label}, "
            f"신뢰도: {result.data.confidence:.3f}"
        )

        # 요청 ID 추가
        result.request_id = request_id
        result.timestamp = datetime.now()

        return result

    except (InvalidImageException, ImageSizeException, UnsupportedImageFormatException) as e:
        logger.warning(f"이미지 검증 실패 - Request ID: {request_id}, 오류: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "message": str(e),
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        ) from e
    except Exception as e:
        logger.error(f"예측 중 오류 발생 - Request ID: {request_id}, 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "예측 처리 중 오류가 발생했습니다",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        ) from e


@router.post("/file", response_model=FilePredictResponse)
async def predict_file(
    file: UploadFile = File(..., description="분석할 이미지 파일"),
    request_id: str = Depends(get_request_id),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    파일 업로드를 통한 예측

    이미지 파일을 직접 업로드하여 불량을 탐지합니다.

    **지원하는 파일 형식:**
    - JPEG (.jpg, .jpeg)
    - PNG (.png)
    - BMP (.bmp)
    - TIFF (.tiff, .tif)

    **파일 크기 제한:**
    - 최대 10MB

    **응답:**
    - 예측된 불량 클래스
    - 신뢰도 점수
    - 파일 정보
    - 처리 시간
    """

    try:
        logger.info(f"파일 예측 요청 시작 - Request ID: {request_id}, 파일명: {file.filename}")

        # 파일 검증
        if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise UnsupportedImageFormatException(
                file.content_type,
                settings.ALLOWED_IMAGE_TYPES
            )

        # 파일 크기 확인
        if file.size is not None and file.size > settings.MAX_IMAGE_SIZE:
            raise ImageSizeException(file.size, settings.MAX_IMAGE_SIZE)
        elif file.size is None:
            # 스트리밍으로 크기만 확인 (메모리 효율적)
            total_size = 0
            chunk_size = 8192
            chunks = []

            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > settings.MAX_IMAGE_SIZE:
                    raise ImageSizeException(total_size, settings.MAX_IMAGE_SIZE)
                chunks.append(chunk)

            # 파일 내용 복원
            await file.seek(0)

        # 예측 수행
        result = await prediction_service.predict_from_file(file)

        logger.info(
            f"파일 예측 완료 - Request ID: {request_id}, "
            f"파일명: {file.filename}, "
            f"결과: {result.data.predicted_label}, "
            f"신뢰도: {result.data.confidence:.3f}"
        )

        # 요청 ID와 파일 정보 추가
        result.request_id = request_id
        result.timestamp = datetime.now()

        return result

    except (InvalidImageException, ImageSizeException, UnsupportedImageFormatException) as e:
        logger.warning(f"파일 검증 실패 - Request ID: {request_id}, 오류: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "message": str(e),
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        ) from e
    except Exception as e:
        logger.error(f"파일 예측 중 오류 발생 - Request ID: {request_id}, 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "파일 예측 처리 중 오류가 발생했습니다",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        ) from e


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(
    files: List[UploadFile] = File(..., description="분석할 이미지 파일들"),
    request_id: str = Depends(get_request_id),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    배치 파일 예측 (선택적 기능)

    여러 이미지 파일을 한 번에 처리합니다.

    **제한사항:**
    - 최대 파일 개수: 10개
    - 각 파일의 크기 제한: 10MB

    **응답:**
    - 각 파일별 예측 결과
    - 전체 처리 통계
    - 실패한 파일 정보
    """

    try:
        logger.info(f"배치 예측 요청 시작 - Request ID: {request_id}, 파일 수: {len(files)}")

        # 배치 크기 제한 확인
        if len(files) > settings.BATCH_SIZE_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "message": f"배치 크기 초과: 최대 {settings.BATCH_SIZE_LIMIT}개 파일까지 처리 가능",
                    "current_count": len(files),
                    "max_count": settings.BATCH_SIZE_LIMIT,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

        # 배치 예측 수행
        result = await prediction_service.predict_batch(files)

        logger.info(
            f"배치 예측 완료 - Request ID: {request_id}, "
            f"성공: {result.data.summary.successful_count}개, "
            f"실패: {result.data.summary.failed_count}개"
        )

        # 요청 ID 추가
        result.request_id = request_id
        result.timestamp = datetime.now()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"배치 예측 중 오류 발생 - Request ID: {request_id}, 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": "배치 예측 처리 중 오류가 발생했습니다",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }
        ) from e