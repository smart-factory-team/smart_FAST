from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone
from ..dependencies import get_model, get_config, get_initialization_status

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    서비스의 기본적인 헬스 상태를 확인합니다.
    주로 로드 밸런서의 Liveness Probe로 사용됩니다.
    """
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/ready")
async def ready_check(
    model_status: None = Depends(get_model), # 모델 로드 여부 체크 (HTTP 500 발생 시 Not Ready)
    config_status: None = Depends(get_config) # 설정 로드 여부 체크
):
    """
    서비스가 요청을 처리할 준비가 되었는지 확인합니다.
    AI 모델과 설정 파일이 정상적으로 로드되었는지 확인합니다.
    주로 로드 밸런서의 Readiness Probe로 사용됩니다.
    """
    return {"status": "ready", "models_loaded": True, "config_loaded": True, "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/startup")
async def startup_check(
    initialization_complete: bool = Depends(get_initialization_status)
):
    """
    애플리케이션의 초기화 과정이 완료되었는지 확인합니다.
    모든 시작 리소스(모델, 설정 등) 로드가 완료되었을 때 'started' 상태를 반환합니다.
    주로 로드 밸런서의 Startup Probe로 사용됩니다.
    """
    if not initialization_complete:
        raise HTTPException(status_code=503, detail="Service initialization not yet complete.")
    return {"status": "started", "initialization_complete": True, "timestamp": datetime.now(timezone.utc).isoformat()}