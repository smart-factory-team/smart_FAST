from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

from app.schemas.common import BaseDataResponse, HealthStatus, ModelStatus


class HealthData(BaseModel):
    """기본 헬스체크 데이터"""
    status: HealthStatus = Field(..., description="헬스 상태")
    uptime_seconds: float = Field(..., ge=0, description="서비스 실행 시간(초)")
    environment: str = Field(..., description="실행 환경")
    version: str = Field(..., description="서비스 버전")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "uptime_seconds": 3600.5,
                "environment": "production",
                "version": "1.0.0"
            }
        }


class HealthResponse(BaseDataResponse):
    """헬스체크 응답"""
    data: HealthData = Field(..., description="헬스체크 데이터")


class ModelInfo(BaseModel):
    """모델 정보"""
    status: ModelStatus = Field(..., description="모델 상태")
    name: Optional[str] = Field(None, description="모델 이름")
    version: Optional[str] = Field(None, description="모델 버전")
    last_used: Optional[datetime] = Field(None, description="마지막 사용 시간")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        json_schema_extra = {
            "example": {
                "status": "loaded",
                "name": "resnet-50-defect-detector",
                "version": "1.0.0",
                "last_used": "2025-08-06T12:00:00Z"
            }
        }


class ReadinessData(BaseModel):
    """준비 상태 데이터 (AI 모델 로딩 상태 체크)"""
    status: str = Field(..., description="준비 상태")
    model: ModelInfo = Field(..., description="모델 정보")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ready",
                "model": {
                    "status": "loaded",
                    "name": "resnet-50-defect-detector",
                    "version": "1.2.0",
                    "last_used": "2024-01-01T12:00:00Z"
                },
                "system": {
                    "cpu_usage": 25.5,
                    "memory_usage": 45.2,
                    "disk_usage": 60.0,
                    "gpu_available": True,
                    "gpu_count": 1
                }
            }
        }


class ReadinessResponse(BaseDataResponse):
    """준비 상태 체크 응답"""
    data: ReadinessData = Field(..., description="준비 상태 데이터")


class StartupCheck(BaseModel):
    """시작 체크 항목"""
    status: str = Field(..., description="체크 상태 (passed/failed/warning/skipped)")
    details: Any = Field(..., description="체크 상세 정보")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "passed",
                "details": "설정 로딩 완료"
            }
        }


class StartupData(BaseModel):
    """시작 준비 데이터 (서비스 시작 준비 체크)"""
    status: str = Field(..., description="시작 준비 상태")
    startup_time: float = Field(..., ge=0, description="시작 소요 시간(초)")
    checks: Dict[str, StartupCheck] = Field(..., description="시작 체크 항목들")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ready",
                "startup_time": 5.2,
                "checks": {
                    "config": {
                        "status": "passed",
                        "details": "환경: production, 버전: 1.0.0"
                    },
                    "directories": {
                        "status": "passed",
                        "details": "모든 필수 디렉토리 존재 확인"
                    },
                    "model": {
                        "status": "passed",
                        "details": "모델 로딩 완료"
                    },
                    "gpu": {
                        "status": "passed",
                        "details": "GPU 사용 가능: 1개 디바이스"
                    }
                }
            }
        }


class StartupResponse(BaseDataResponse):
    """시작 준비 체크 응답"""
    data: StartupData = Field(..., description="시작 준비 데이터")