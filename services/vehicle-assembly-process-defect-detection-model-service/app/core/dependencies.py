"""
FastAPI 의존성 주입 모듈

애플리케이션 전반에서 사용되는 공통 의존성들을 정의합니다.
- 요청 ID 생성
- 모델 매니저 인스턴스 제공
- 인증 및 권한 관리 (필요시)
- 데이터베이스 연결 (필요시)
"""

import uuid
import logging
from typing import Optional, TYPE_CHECKING
from fastapi import Depends, HTTPException, status, Request
from datetime import datetime

from app.core.config import settings

# 로거 설정
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.services.model_manager import DefectDetectionModelManager

# 전역 모델 매니저 인스턴스
_model_manager: Optional[DefectDetectionModelManager] = None


async def get_request_id() -> str:
    """
    고유한 요청 ID 생성

    각 API 요청에 대해 고유한 ID를 생성하여 로깅 및 추적에 사용합니다.
    UUID4를 사용하여 충돌 가능성을 최소화합니다.

    Returns:
        str: 고유한 요청 ID (UUID4 형식)
    """
    request_id = str(uuid.uuid4())
    logger.debug(f"새 요청 ID 생성: {request_id}")
    return request_id


async def get_model_manager() -> 'DefectDetectionModelManager':
    """
    모델 매니저 인스턴스 제공

    싱글톤 패턴으로 모델 매니저를 관리합니다.
    앱 시작 시 한 번만 초기화되고, 모든 요청에서 동일한 인스턴스를 사용합니다.

    Returns:
        DefectDetectionModelManager: 모델 매니저 인스턴스

    Raises:
        HTTPException: 모델 매니저가 초기화되지 않은 경우
    """
    global _model_manager

    logger.info("=== get_model_manager 호출됨 ===")

    if _model_manager is None:
        logger.error("모델 매니저가 초기화되지 않았습니다")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "success": False,
                "message": "모델 매니저가 초기화되지 않았습니다. 서비스를 재시작해주세요.",
                "error_code": "MODEL_MANAGER_NOT_INITIALIZED",
                "timestamp": datetime.now().isoformat()
            }
        )

    try:
        is_loaded = _model_manager.is_loaded
        logger.info(f"모델 로드 상태: {is_loaded}")
    except Exception as e:
        logger.error(f"is_loaded 접근 중 오류: {e}")
        raise HTTPException(...)

    if not is_loaded:
        logger.warning("모델이 로드되지 않았습니다")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "success": False,
                "message": "AI 모델이 로드되지 않았습니다. 잠시 후 다시 시도해주세요.",
                "error_code": "MODEL_NOT_LOADED",
                "timestamp": datetime.now().isoformat()
            }
        )

    logger.info("모델 매니저 반환")
    return _model_manager


async def set_model_manager(model_manager: 'DefectDetectionModelManager') -> None:
    """
    모델 매니저 인스턴스 설정

    앱 시작 시점에 호출되어 전역 모델 매니저를 설정합니다.
    main.py의 lifespan 이벤트에서 사용됩니다.

    Args:
        model_manager: 설정할 모델 매니저 인스턴스
    """
    global _model_manager
    _model_manager = model_manager
    logger.info("모델 매니저 의존성 설정 완료")


async def get_model_manager_optional() -> Optional['DefectDetectionModelManager']:
    """
    모델 매니저 인스턴스 제공 (선택적)

    모델 매니저가 없어도 오류를 발생시키지 않는 버전입니다.
    헬스체크나 상태 확인 시 사용됩니다.

    Returns:
        Optional[DefectDetectionModelManager]: 모델 매니저 인스턴스 또는 None
    """
    global _model_manager
    return _model_manager


def get_client_ip(request: Request) -> str:
    """
    클라이언트 IP 주소 추출

    프록시나 로드밸런서를 고려하여 실제 클라이언트 IP를 추출합니다.
    로깅 및 모니터링 목적으로 사용됩니다.

    Args:
        request: FastAPI Request 객체

    Returns:
        str: 클라이언트 IP 주소
    """
    # X-Forwarded-For 헤더 확인 (프록시/로드밸런서 뒤에 있는 경우)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # 첫 번째 IP가 실제 클라이언트 IP
        return forwarded_for.split(",")[0].strip()

    # X-Real-IP 헤더 확인
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # 기본값으로 request.client.host 사용
    return request.client.host if request.client else "unknown"


def get_user_agent(request: Request) -> str:
    """
    User-Agent 정보 추출

    클라이언트의 User-Agent 정보를 추출합니다.
    API 사용 통계 및 모니터링에 활용됩니다.

    Args:
        request: FastAPI Request 객체

    Returns:
        str: User-Agent 문자열
    """
    return request.headers.get("User-Agent", "unknown")


async def verify_api_key(request: Request) -> bool:
    """
    API 키 검증 (선택적 기능)

    설정에서 API 키 인증이 활성화된 경우 요청의 API 키를 검증합니다.
    프로덕션 환경에서 보안 강화를 위해 사용할 수 있습니다.

    Args:
        request: FastAPI Request 객체

    Returns:
        bool: API 키 유효성

    Raises:
        HTTPException: API 키가 유효하지 않은 경우
    """
    # API 키 인증이 비활성화된 경우 통과
    if not settings.REQUIRE_API_KEY:
        return True

    # Authorization 헤더 또는 X-API-Key 헤더에서 키 추출
    auth_header = request.headers.get("Authorization")
    api_key = request.headers.get("X-API-Key")

    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header[7:]  # "Bearer " 제거

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "message": "API 키가 필요합니다",
                "error_code": "API_KEY_MISSING",
                "timestamp": datetime.now().isoformat()
            }
        )

    # API 키 검증 (실제 운영에서는 데이터베이스나 외부 서비스로 검증)
    if api_key != settings.API_KEY:
        logger.warning(f"유효하지 않은 API 키 시도: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "message": "유효하지 않은 API 키입니다",
                "error_code": "INVALID_API_KEY",
                "timestamp": datetime.now().isoformat()
            }
        )

    return True


class RequestContext:
    """
    요청 컨텍스트 정보를 담는 클래스

    각 요청에 대한 메타데이터를 수집하고 관리합니다.
    로깅, 모니터링, 디버깅에 활용됩니다.
    """

    def __init__(
        self,
        request_id: str,
        client_ip: str,
        user_agent: str,
        timestamp: datetime,
        endpoint: str,
        method: str
    ):
        self.request_id = request_id
        self.client_ip = client_ip
        self.user_agent = user_agent
        self.timestamp = timestamp
        self.endpoint = endpoint
        self.method = method
        self.processing_start = datetime.now()

    def get_processing_duration(self) -> float:
        """요청 처리 시간 계산 (초)"""
        return (datetime.now() - self.processing_start).total_seconds()

    def to_dict(self) -> dict:
        """딕셔너리 형태로 변환"""
        return {
            "request_id": self.request_id,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "timestamp": self.timestamp.isoformat(),
            "endpoint": self.endpoint,
            "method": self.method,
            "processing_duration": self.get_processing_duration()
        }


async def get_request_context(
    request: Request,
    request_id: str = Depends(get_request_id)
) -> RequestContext:
    """
    요청 컨텍스트 생성

    각 요청에 대한 상세 정보를 수집하여 RequestContext 객체를 생성합니다.

    Args:
        request: FastAPI Request 객체
        request_id: 요청 ID (의존성에서 자동 주입)

    Returns:
        RequestContext: 요청 컨텍스트 객체
    """
    return RequestContext(
        request_id=request_id,
        client_ip=get_client_ip(request),
        user_agent=get_user_agent(request),
        timestamp=datetime.now(),
        endpoint=str(request.url.path),
        method=request.method
    )


# 관리자 권한 검증 의존성 (필요시 사용)
async def verify_admin_access(
    request: Request,
    api_key_valid: bool = Depends(verify_api_key)
) -> bool:
    """
    관리자 권한 검증

    모델 재로딩, 시스템 설정 변경 등 관리자 기능에 대한 접근을 제한합니다.

    Args:
        request: FastAPI Request 객체
        api_key_valid: API 키 유효성 (의존성에서 자동 검증)

    Returns:
        bool: 관리자 권한 유효성

    Raises:
        HTTPException: 관리자 권한이 없는 경우
    """
    # 개발 환경에서는 관리자 권한 검증 생략
    if settings.ENVIRONMENT == "development":
        return True

    # 관리자 전용 API 키 또는 토큰 검증
    admin_token = request.headers.get("X-Admin-Token")

    if not admin_token or admin_token != settings.ADMIN_TOKEN:
        logger.warning(f"관리자 권한 없는 접근 시도: {get_client_ip(request)}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "success": False,
                "message": "관리자 권한이 필요합니다",
                "error_code": "ADMIN_ACCESS_REQUIRED",
                "timestamp": datetime.now().isoformat()
            }
        )

    return True


# 의존성 정리 함수
async def cleanup_dependencies():
    """
    애플리케이션 종료 시 의존성 리소스 정리

    전역 변수들을 정리하고 리소스를 해제합니다.
    main.py의 lifespan 이벤트에서 호출됩니다.
    """
    global _model_manager

    try:
        if _model_manager:
            await _model_manager.cleanup()
            _model_manager = None
            logger.info("모델 매니저 의존성 정리 완료")

    except Exception as e:
        logger.error(f"의존성 정리 중 오류 발생: {e}")


# 애플리케이션 상태 확인 헬퍼
def get_application_health() -> dict:
    """
    애플리케이션 전체 상태 확인

    Returns:
        dict: 애플리케이션 상태 정보
    """
    global _model_manager

    status = {
        "model_manager_initialized": _model_manager is not None,
        "model_loaded": _model_manager.is_loaded if _model_manager else False,
        "environment": settings.ENVIRONMENT,
        "debug_mode": settings.DEBUG
    }

    if _model_manager:
        status.update({
            "model_status": _model_manager.status.value,
            "model_name": _model_manager.model_name,
            "model_version": _model_manager.model_version,
            "total_predictions": _model_manager.statistics.total_predictions
        })

    return status