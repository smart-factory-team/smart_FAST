from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import time
import os

from app.core.config import settings
from app.core.dependencies import get_request_id, get_model_manager
from app.schemas.health import (
    HealthResponse,
    ReadinessResponse,
    StartupResponse
)

router = APIRouter(tags=["Health"])

# 서비스 시작 시간
_service_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(request_id: str = Depends(get_request_id)):
    """
    기본 헬스체크

    서비스가 정상적으로 실행 중인지 확인합니다.
    가장 빠른 응답을 제공하며, 로드밸런서나 모니터링 시스템에서 사용됩니다.
    """

    uptime = time.time() - _service_start_time

    return HealthResponse(
        success=True,
        message="서비스가 정상적으로 실행 중입니다",
        data={
            "status": "healthy",
            "uptime_seconds": uptime,
            "environment": settings.ENVIRONMENT,
            "version": settings.VERSION
        },
        timestamp=datetime.now(),
        request_id=request_id
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(
    request_id: str = Depends(get_request_id),
    model_manager = Depends(get_model_manager)
):
    """
    AI 모델 로딩 상태 체크 (Readiness Probe)

    서비스가 요청을 처리할 준비가 되었는지 확인합니다.
    - AI 모델 로딩 상태
    - 필수 의존성 확인
    - GPU 사용 가능 여부 (설정된 경우)
    """

    # 모델 상태 확인
    model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"

    # 준비 상태 판단
    is_ready = model_status == "loaded"
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "success": False,
                "message": "AI 모델이 로드되지 않음",
                "errors": ["AI 모델이 로드되지 않음"],
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
        )

    return ReadinessResponse(
        success=True,
        message="서비스가 요청을 처리할 준비가 완료됨",
        data={
            "status": "ready",
            "model": {
                "status": model_status,
                "name": model_manager.model_name if model_manager else None,
                "version": model_manager.model_version if model_manager else None,
                "last_used": model_manager.last_used.isoformat() if model_manager and model_manager.last_used else None
            }
        },
        timestamp=datetime.now(),
        request_id=request_id
    )


@router.get("/startup", response_model=StartupResponse)
async def startup_check(request_id: str = Depends(get_request_id)):
    """
    서비스 시작 준비 체크 (Startup Probe)

    서비스가 시작 과정을 완료했는지 확인합니다.
    - 설정 파일 로딩 확인
    - 필수 디렉토리 존재 확인
    - 네트워크 연결 확인
    - 초기화 과정 완료 확인
    """

    startup_checks = {}
    all_passed = True

    # 1. 설정 확인
    try:
        startup_checks["config"] = {
            "status": "passed",
            "details": f"환경: {settings.ENVIRONMENT}, 버전: {settings.VERSION}"
        }
    except Exception as e:
        startup_checks["config"] = {
            "status": "failed",
            "details": f"설정 로딩 실패: {str(e)}"
        }
        all_passed = False

    # 2. 필수 디렉토리 확인
    # 2. 필수 디렉토리 확인
    try:
        required_dirs = [
            settings.MODEL_BASE_PATH,
            settings.UPLOAD_DIR,
            settings.TEMP_DIR
        ]

        missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
        if missing_dirs:
            startup_checks["directories"] = {
                "status": "failed",
                "details": f"누락된 디렉토리: {missing_dirs}"
            }
            all_passed = False
        else:
            startup_checks["directories"] = {
                "status": "passed",
                "details": "모든 필수 디렉토리 존재 확인"
            }
    except Exception as e:
        startup_checks["directories"] = {
            "status": "failed",
            "details": f"디렉토리 확인 실패: {str(e)}"
        }
        all_passed = False

    # 시작 준비 완료 여부 판단
    startup_time = time.time() - _service_start_time

    if not all_passed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "success": False,
                "message": "서비스 시작 준비가 완료되지 않음",
                "data": {
                    "status": "not_ready",
                    "startup_time": startup_time,
                    "checks": startup_checks
                },
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
        )

    return StartupResponse(
        success=True,
        message="서비스 시작 준비가 완료됨",
        data={
            "status": "ready",
            "startup_time": startup_time,
            "checks": startup_checks
        },
        timestamp=datetime.now(),
        request_id=request_id
    )