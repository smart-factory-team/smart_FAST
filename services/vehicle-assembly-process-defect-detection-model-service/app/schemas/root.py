from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from app.schemas.common import BaseDataResponse


class ApiDocs(BaseModel):
    """API 문서 링크"""
    swagger_ui: Optional[str] = Field(None, description="Swagger UI 링크")
    redoc: Optional[str] = Field(None, description="ReDoc 링크")
    openapi_json: Optional[str] = Field(None, description="OpenAPI JSON 스키마")

    class Config:
        json_schema_extra = {
            "example": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_json": "/api/v1/openapi.json"
            }
        }


class ServiceEndpoints(BaseModel):
    """서비스 엔드포인트 목록"""
    health_check: str = Field(..., description="헬스체크")
    readiness_check: str = Field(..., description="준비 상태 체크")
    startup_check: str = Field(..., description="시작 준비 체크")
    prediction: str = Field(..., description="AI 모델 예측")
    file_prediction: str = Field(..., description="파일 업로드 예측")
    model_info: str = Field(..., description="모델 정보 조회")

    class Config:
        json_schema_extra = {
            "example": {
                "health_check": "/health",
                "readiness_check": "/ready",
                "startup_check": "/startup",
                "prediction": "/predict",
                "file_prediction": "/predict/file",
                "model_info": "/model/info"
            }
        }


class ServiceInfo(BaseModel):
    """서비스 정보 데이터"""
    service_name: str = Field(..., description="서비스 이름")
    version: str = Field(..., description="서비스 버전")
    description: str = Field(..., description="서비스 설명")
    environment: str = Field(..., description="실행 환경")
    status: str = Field(..., description="서비스 상태")
    api_docs: ApiDocs = Field(..., description="API 문서 링크")
    endpoints: ServiceEndpoints = Field(..., description="사용 가능한 엔드포인트")
    features: List[str] = Field(..., description="주요 기능 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "service_name": "자동차 의장 공정 불량 탐지 API",
                "version": "1.0.0",
                "description": "자동차 의장 공정에서 발생하는 불량을 탐지하는 이미지 분류 API",
                "environment": "production",
                "status": "running",
                "api_docs": {
                    "swagger_ui": "/docs",
                    "redoc": "/redoc",
                    "openapi_json": "/api/v1/openapi.json"
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
            }
        }


class ServiceInfoResponse(BaseDataResponse):
    """서비스 정보 응답"""
    data: ServiceInfo = Field(..., description="서비스 정보")