"""
커스텀 예외 처리 모듈

애플리케이션에서 발생하는 다양한 예외들을 정의하고 처리합니다.
- 이미지 관련 예외
- 모델 관련 예외
- API 관련 예외
- 전역 예외 핸들러
"""

import logging
from typing import Any, Dict
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)


# === 커스텀 예외 클래스들 ===

class BaseAPIException(Exception):
    """API 예외 기본 클래스"""

    def __init__(
        self,
        message: str,
        error_code: str = "API_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Dict[str, Any] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class InvalidImageException(BaseAPIException):
    """유효하지 않은 이미지 예외"""

    def __init__(self, message: str = "유효하지 않은 이미지입니다"):
        super().__init__(
            message=message,
            error_code="INVALID_IMAGE",
            status_code=status.HTTP_400_BAD_REQUEST
        )


class UnsupportedImageFormatException(BaseAPIException):
    """지원하지 않는 이미지 형식 예외"""

    def __init__(self, provided_format: str, supported_formats: list):
        message = f"지원하지 않는 이미지 형식입니다: {provided_format}"
        details = {
            "provided_format": provided_format,
            "supported_formats": supported_formats
        }
        super().__init__(
            message=message,
            error_code="UNSUPPORTED_IMAGE_FORMAT",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class ImageSizeException(BaseAPIException):
    """이미지 크기 초과 예외"""

    def __init__(self, actual_size: int, max_size: int):
        if isinstance(actual_size, int) and isinstance(max_size, int):
            # 바이트 크기
            actual_mb = actual_size / (1024 * 1024)
            max_mb = max_size / (1024 * 1024)
            message = f"이미지 크기가 초과되었습니다: {actual_mb:.1f}MB (최대: {max_mb:.1f}MB)"
        else:
            # 차원 크기
            message = f"이미지 차원이 초과되었습니다: {actual_size} (최대: {max_size})"

        details = {
            "actual_size": actual_size,
            "max_size": max_size
        }
        super().__init__(
            message=message,
            error_code="IMAGE_SIZE_EXCEEDED",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            details=details
        )


class ModelNotLoadedException(BaseAPIException):
    """모델이 로드되지 않음 예외"""

    def __init__(self, message: str = "AI 모델이 로드되지 않았습니다"):
        super().__init__(
            message=message,
            error_code="MODEL_NOT_LOADED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class ModelPredictionException(BaseAPIException):
    """모델 예측 실패 예외"""

    def __init__(self, message: str = "모델 예측 중 오류가 발생했습니다", details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="MODEL_PREDICTION_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class ModelTimeoutException(BaseAPIException):
    """모델 예측 타임아웃 예외"""

    def __init__(self, timeout_seconds: int):
        message = f"모델 예측 시간 초과: {timeout_seconds}초"
        details = {"timeout_seconds": timeout_seconds}
        super().__init__(
            message=message,
            error_code="MODEL_PREDICTION_TIMEOUT",
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            details=details
        )


class BatchSizeLimitException(BaseAPIException):
    """배치 크기 제한 초과 예외"""

    def __init__(self, actual_size: int, max_size: int):
        message = f"배치 크기 초과: {actual_size}개 (최대: {max_size}개)"
        details = {
            "actual_size": actual_size,
            "max_size": max_size
        }
        super().__init__(
            message=message,
            error_code="BATCH_SIZE_EXCEEDED",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class APIKeyMissingException(BaseAPIException):
    """API 키 누락 예외"""

    def __init__(self):
        super().__init__(
            message="API 키가 필요합니다",
            error_code="API_KEY_MISSING",
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class InvalidAPIKeyException(BaseAPIException):
    """유효하지 않은 API 키 예외"""

    def __init__(self):
        super().__init__(
            message="유효하지 않은 API 키입니다",
            error_code="INVALID_API_KEY",
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class AdminAccessRequiredException(BaseAPIException):
    """관리자 권한 필요 예외"""

    def __init__(self):
        super().__init__(
            message="관리자 권한이 필요합니다",
            error_code="ADMIN_ACCESS_REQUIRED",
            status_code=status.HTTP_403_FORBIDDEN
        )


class ResourceNotFoundException(BaseAPIException):
    """리소스를 찾을 수 없음 예외"""

    def __init__(self, resource_type: str, resource_id: str = None):
        if resource_id:
            message = f"{resource_type}을(를) 찾을 수 없습니다: {resource_id}"
        else:
            message = f"{resource_type}을(를) 찾을 수 없습니다"

        details = {
            "resource_type": resource_type,
            "resource_id": resource_id
        }
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class RateLimitExceededException(BaseAPIException):
    """요청 제한 초과 예외"""

    def __init__(self, limit: int, window_seconds: int):
        message = f"요청 제한 초과: {limit}회/{window_seconds}초"
        details = {
            "limit": limit,
            "window_seconds": window_seconds
        }
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


# === 예외 핸들러들 ===

async def base_api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """기본 API 예외 핸들러"""

    error_response = {
        "success": False,
        "message": exc.message,
        "error_code": exc.error_code,
        "timestamp": datetime.now().isoformat(),
        "path": str(request.url.path),
        "method": request.method
    }

    # 세부 정보 추가 (있는 경우)
    if exc.details:
        error_response["details"] = exc.details

    # 요청 ID 추가 (있는 경우)
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        error_response["request_id"] = request_id

    # 로깅
    log_level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    logger.log(
        log_level,
        f"API 예외 발생: {exc.error_code} - {exc.message} "
        f"(상태코드: {exc.status_code}, 경로: {request.url.path})"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP 예외 핸들러"""

    error_response = {
        "success": False,
        "message": exc.detail if isinstance(exc.detail, str) else "HTTP 오류가 발생했습니다",
        "error_code": f"HTTP_{exc.status_code}",
        "timestamp": datetime.now().isoformat(),
        "path": str(request.url.path),
        "method": request.method
    }

    # 세부 정보가 딕셔너리인 경우 병합
    if isinstance(exc.detail, dict):
        error_response.update(exc.detail)

    # 요청 ID 추가
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        error_response["request_id"] = request_id

    # 로깅
    log_level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    logger.log(
        log_level,
        f"HTTP 예외 발생: {exc.status_code} - {error_response['message']} "
        f"(경로: {request.url.path})"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """요청 검증 예외 핸들러"""

    # 검증 오류 상세 정보 구성
    validation_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        validation_errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })

    error_response = {
        "success": False,
        "message": "요청 데이터 검증에 실패했습니다",
        "error_code": "VALIDATION_ERROR",
        "details": {
            "validation_errors": validation_errors
        },
        "timestamp": datetime.now().isoformat(),
        "path": str(request.url.path),
        "method": request.method
    }

    # 요청 ID 추가
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        error_response["request_id"] = request_id

    # 로깅
    logger.warning(
        f"요청 검증 실패: {len(validation_errors)}개 오류 "
        f"(경로: {request.url.path})"
    )

    # JSON 직렬화 불가능한 객체들을 문자열로 변환
    def clean_for_json(obj):
        if isinstance(obj, bytes):
            return f"<bytes object: {len(obj)} bytes>"
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        else:
            return obj

    # 에러 세부사항 정리
    cleaned_errors = clean_for_json(exc.errors())

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """일반 예외 핸들러 (예상치 못한 오류)"""

    error_response = {
        "success": False,
        "message": "서버 내부 오류가 발생했습니다",
        "error_code": "INTERNAL_SERVER_ERROR",
        "timestamp": datetime.now().isoformat(),
        "path": str(request.url.path),
        "method": request.method
    }

    # 요청 ID 추가
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        error_response["request_id"] = request_id

    # 개발 환경에서만 상세 오류 정보 노출
    from app.core.config import settings
    if settings.DEBUG:
        error_response["details"] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)
        }

    # 로깅 (전체 스택 트레이스 포함)
    logger.error(
        f"예상치 못한 예외 발생: {type(exc).__name__} - {str(exc)} "
        f"(경로: {request.url.path})",
        exc_info=True
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )


# === 예외 핸들러 설정 함수 ===

def setup_exception_handlers(app: FastAPI) -> None:
    """
    FastAPI 앱에 예외 핸들러들을 등록

    Args:
        app: FastAPI 인스턴스
    """

    # 커스텀 API 예외 핸들러
    app.add_exception_handler(BaseAPIException, base_api_exception_handler)

    # FastAPI HTTP 예외 핸들러
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # 요청 검증 예외 핸들러
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # 일반 예외 핸들러 (마지막에 등록 - 최종 fallback)
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("예외 핸들러 설정 완료")


# === 예외 발생 헬퍼 함수들 ===

def raise_invalid_image(message: str = None) -> None:
    """유효하지 않은 이미지 예외 발생"""
    raise InvalidImageException(message or "유효하지 않은 이미지입니다")


def raise_unsupported_format(provided_format: str, supported_formats: list) -> None:
    """지원하지 않는 형식 예외 발생"""
    raise UnsupportedImageFormatException(provided_format, supported_formats)


def raise_image_size_exceeded(actual_size: int, max_size: int) -> None:
    """이미지 크기 초과 예외 발생"""
    raise ImageSizeException(actual_size, max_size)


def raise_model_not_loaded() -> None:
    """모델 미로드 예외 발생"""
    raise ModelNotLoadedException()


def raise_prediction_failed(message: str = None, details: Dict[str, Any] = None) -> None:
    """예측 실패 예외 발생"""
    raise ModelPredictionException(message, details)


def raise_batch_size_exceeded(actual_size: int, max_size: int) -> None:
    """배치 크기 초과 예외 발생"""
    raise BatchSizeLimitException(actual_size, max_size)


def raise_resource_not_found(resource_type: str, resource_id: str = None) -> None:
    """리소스 미발견 예외 발생"""
    raise ResourceNotFoundException(resource_type, resource_id)


def raise_api_key_missing() -> None:
    """API 키 누락 예외 발생"""
    raise APIKeyMissingException()


def raise_invalid_api_key() -> None:
    """유효하지 않은 API 키 예외 발생"""
    raise InvalidAPIKeyException()


def raise_admin_access_required() -> None:
    """관리자 권한 필요 예외 발생"""
    raise AdminAccessRequiredException()


def raise_rate_limit_exceeded(limit: int, window_seconds: int) -> None:
    """요청 제한 초과 예외 발생"""
    raise RateLimitExceededException(limit, window_seconds)


# === 예외 변환 헬퍼 함수들 ===

def convert_pil_error_to_api_exception(error: Exception) -> BaseAPIException:
    """
    PIL 라이브러리 오류를 API 예외로 변환

    Args:
        error: PIL에서 발생한 예외

    Returns:
        BaseAPIException: 변환된 API 예외
    """
    error_message = str(error).lower()

    if "cannot identify image file" in error_message:
        return InvalidImageException("이미지 파일을 인식할 수 없습니다")
    elif "image file is truncated" in error_message:
        return InvalidImageException("이미지 파일이 손상되었습니다")
    elif "decoder" in error_message:
        return InvalidImageException("이미지 디코딩 중 오류가 발생했습니다")
    else:
        return InvalidImageException(f"이미지 처리 중 오류가 발생했습니다: {str(error)}")


def convert_torch_error_to_api_exception(error: Exception) -> BaseAPIException:
    """
    PyTorch 오류를 API 예외로 변환

    Args:
        error: PyTorch에서 발생한 예외

    Returns:
        BaseAPIException: 변환된 API 예외
    """
    error_message = str(error).lower()

    if "cuda" in error_message and "out of memory" in error_message:
        return ModelPredictionException(
            "GPU 메모리 부족으로 예측에 실패했습니다",
            {"error_type": "gpu_out_of_memory"}
        )
    elif "cuda" in error_message:
        return ModelPredictionException(
            "GPU 처리 중 오류가 발생했습니다",
            {"error_type": "gpu_error"}
        )
    elif "size mismatch" in error_message:
        return ModelPredictionException(
            "입력 데이터 크기가 모델과 호환되지 않습니다",
            {"error_type": "input_size_mismatch"}
        )
    else:
        return ModelPredictionException(
            f"모델 처리 중 오류가 발생했습니다: {str(error)}",
            {"error_type": "torch_error"}
        )


def convert_requests_error_to_api_exception(error: Exception) -> BaseAPIException:
    """
    HTTP 요청 오류를 API 예외로 변환

    Args:
        error: requests 라이브러리에서 발생한 예외

    Returns:
        BaseAPIException: 변환된 API 예외
    """
    error_message = str(error).lower()

    if "timeout" in error_message:
        return InvalidImageException("이미지 다운로드 시간 초과")
    elif "connection" in error_message:
        return InvalidImageException("이미지 다운로드 연결 실패")
    elif "404" in error_message:
        return InvalidImageException("이미지를 찾을 수 없습니다")
    else:
        return InvalidImageException(f"이미지 다운로드 실패: {str(error)}")


# === 예외 로깅 헬퍼 ===

def log_exception_with_context(
    exc: Exception,
    request: Request = None,
    extra_context: Dict[str, Any] = None
) -> None:
    """
    예외를 컨텍스트 정보와 함께 로깅

    Args:
        exc: 발생한 예외
        request: FastAPI Request 객체 (선택적)
        extra_context: 추가 컨텍스트 정보 (선택적)
    """
    context = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "timestamp": datetime.now().isoformat()
    }

    if request:
        context.update({
            "method": request.method,
            "path": str(request.url.path),
            "query_params": str(request.query_params),
            "client_host": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("User-Agent", "unknown")
        })

        # 요청 ID 추가
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            context["request_id"] = request_id

    if extra_context:
        context.update(extra_context)

    logger.error(
        f"예외 발생: {context['exception_type']} - {context['exception_message']}",
        extra=context,
        exc_info=True
    )


# === 상태 코드 매핑 ===

ERROR_CODE_TO_STATUS_CODE = {
    "INVALID_IMAGE": status.HTTP_400_BAD_REQUEST,
    "UNSUPPORTED_IMAGE_FORMAT": status.HTTP_400_BAD_REQUEST,
    "IMAGE_SIZE_EXCEEDED": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    "MODEL_NOT_LOADED": status.HTTP_503_SERVICE_UNAVAILABLE,
    "MODEL_PREDICTION_FAILED": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "MODEL_PREDICTION_TIMEOUT": status.HTTP_408_REQUEST_TIMEOUT,
    "BATCH_SIZE_EXCEEDED": status.HTTP_400_BAD_REQUEST,
    "API_KEY_MISSING": status.HTTP_401_UNAUTHORIZED,
    "INVALID_API_KEY": status.HTTP_401_UNAUTHORIZED,
    "ADMIN_ACCESS_REQUIRED": status.HTTP_403_FORBIDDEN,
    "RESOURCE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "RATE_LIMIT_EXCEEDED": status.HTTP_429_TOO_MANY_REQUESTS,
    "VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "INTERNAL_SERVER_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR
}


def get_status_code_for_error(error_code: str) -> int:
    """
    에러 코드에 해당하는 HTTP 상태 코드 반환

    Args:
        error_code: 에러 코드

    Returns:
        int: HTTP 상태 코드
    """
    return ERROR_CODE_TO_STATUS_CODE.get(error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)


# === 예외 통계 (선택적) ===

class ExceptionStats:
    """예외 발생 통계를 관리하는 클래스"""

    def __init__(self):
        self.exception_counts = {}
        self.last_reset = datetime.now()

    def record_exception(self, exception_type: str, error_code: str = None):
        """예외 발생 기록"""
        key = f"{exception_type}:{error_code}" if error_code else exception_type
        self.exception_counts[key] = self.exception_counts.get(key, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            "exception_counts": self.exception_counts,
            "total_exceptions": sum(self.exception_counts.values()),
            "unique_exception_types": len(self.exception_counts),
            "stats_since": self.last_reset.isoformat()
        }

    def reset_stats(self):
        """통계 초기화"""
        self.exception_counts.clear()
        self.last_reset = datetime.now()


# 전역 예외 통계 인스턴스
exception_stats = ExceptionStats()


def record_exception_stat(exc: Exception, error_code: str = None):
    """예외 통계 기록"""
    exception_type = type(exc).__name__
    exception_stats.record_exception(exception_type, error_code)


def get_exception_statistics() -> Dict[str, Any]:
    """예외 통계 조회"""
    return exception_stats.get_stats()


def reset_exception_statistics():
    """예외 통계 초기화"""
    exception_stats.reset_stats()