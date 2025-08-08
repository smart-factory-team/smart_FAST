from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

from app.schemas.common import (
    BaseDataResponse,
    DefectCategory,
    ModelStatus,
    DEFECT_CATEGORIES
)


class ModelConfiguration(BaseModel):
    """모델 설정 정보"""
    input_size: List[int] = Field(..., description="입력 이미지 크기 [width, height]")
    num_classes: int = Field(..., ge=1, description="분류 클래스 수")
    architecture: str = Field(..., description="모델 아키텍처")
    model_source: str = Field(..., description="모델 출처 (HuggingFace 등)")

    class Config:
        json_schema_extra = {
            "example": {
                "input_size": [224, 224],
                "num_classes": 13,
                "architecture": "ResNet-50",
                "model_source": "huggingface"
            }
        }


class ModelStatistics(BaseModel):
    """모델 사용 통계"""
    total_predictions: int = Field(default=0, ge=0, description="총 예측 횟수")
    successful_predictions: int = Field(default=0, ge=0, description="성공한 예측 횟수")
    failed_predictions: int = Field(default=0, ge=0, description="실패한 예측 횟수")
    average_processing_time: Optional[float] = Field(None, ge=0, description="평균 처리 시간(초)")
    last_used: Optional[datetime] = Field(None, description="마지막 사용 시간")
    uptime_hours: Optional[float] = Field(None, ge=0, description="가동 시간(시간)")
    class_prediction_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="클래스별 예측 횟수"
    )
    error_rate: Optional[float] = Field(None, ge=0, le=1, description="에러율")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        json_schema_extra = {
            "example": {
                "total_predictions": 1250,
                "successful_predictions": 1235,
                "failed_predictions": 15,
                "average_processing_time": 0.145,
                "last_used": "2024-01-01T15:30:00Z",
                "uptime_hours": 72.5,
                "class_prediction_counts": {
                    "정상": 800,
                    "스크래치": 200,
                    "고정 불량": 150,
                    "외관 손상": 85
                },
                "error_rate": 0.012
            }
        }


class ModelVersionInfo(BaseModel):
    """모델 버전 정보"""
    version: str = Field(..., description="모델 버전")
    release_date: Optional[datetime] = Field(None, description="릴리스 일시")
    description: Optional[str] = Field(None, description="버전 설명")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        json_schema_extra = {
            "example": {
                "version": "1.2.0",
                "release_date": "2024-01-15T00:00:00Z",
                "description": "정확도 개선 및 새로운 불량 타입 추가"
            }
        }


class PerformanceMetrics(BaseModel):
    """모델 성능 지표"""
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="정확도")
    precision: Optional[float] = Field(None, ge=0, le=1, description="정밀도")
    recall: Optional[float] = Field(None, ge=0, le=1, description="재현율")
    f1_score: Optional[float] = Field(None, ge=0, le=1, description="F1 점수")
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="혼동 행렬")
    class_wise_performance: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="클래스별 성능"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.89,
                "f1_score": 0.90,
                "confusion_matrix": [
                    [950, 30, 20],
                    [15, 285, 0],
                    [5, 10, 125]
                ],
                "class_wise_performance": {
                    "정상": {"precision": 0.98, "recall": 0.95, "f1_score": 0.96},
                    "스크래치": {"precision": 0.90, "recall": 0.88, "f1_score": 0.89},
                    "고정 불량": {"precision": 0.85, "recall": 0.92, "f1_score": 0.88}
                }
            }
        }


class ModelInfo(BaseModel):
    """모델 상세 정보"""
    # 기본 정보
    name: str = Field(..., description="모델 이름")
    status: ModelStatus = Field(..., description="모델 상태")
    version_info: ModelVersionInfo = Field(..., description="버전 정보")

    # 설정 및 구성
    configuration: ModelConfiguration = Field(..., description="모델 설정")
    supported_categories: List[DefectCategory] = Field(..., description="지원하는 불량 카테고리")

    # 사용 통계
    usage_statistics: ModelStatistics = Field(..., description="사용 통계")

    # 메타데이터
    created_date: Optional[datetime] = Field(None, description="모델 생성 일시")
    loaded_date: Optional[datetime] = Field(None, description="모델 로드 일시")
    model_size_mb: Optional[float] = Field(None, gt=0, description="모델 파일 크기(MB)")
    description: Optional[str] = Field(None, description="모델 설명")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        json_schema_extra = {
            "example": {
                "name": "ResNet50-CarDefectDetector",
                "status": "loaded",
                "version_info": {
                    "version": "1.2.0",
                    "release_date": "2024-01-15T00:00:00Z",
                    "description": "정확도 개선 및 새로운 불량 타입 추가"
                },
                "configuration": {
                    "input_size": [224, 224],
                    "num_classes": 13,
                    "architecture": "ResNet-50",
                    "model_source": "huggingface"
                },
                "supported_categories": [
                    {"id": 0, "name": "정상", "supercategory": "품질상태"},
                    {"id": 101, "name": "스크래치", "supercategory": "자동차부품품질"},
                    {"id": 102, "name": "외관 손상", "supercategory": "자동차부품품질"}
                ],
                "usage_statistics": {
                    "total_predictions": 1250,
                    "successful_predictions": 1235,
                    "average_processing_time": 0.145,
                    "last_used": "2024-01-01T15:30:00Z"
                },
                "created_date": "2024-01-01T00:00:00Z",
                "loaded_date": "2024-01-01T09:00:00Z",
                "model_size_mb": 102.5,
                "description": "자동차 부품 품질 검사를 위한 ResNet-50 기반 분류 모델"
            }
        }


class ModelInfoResponse(BaseDataResponse):
    """모델 정보 조회 응답 (/model/info)"""
    data: ModelInfo = Field(..., description="모델 정보")


class ModelClassesInfo(BaseModel):
    """모델 클래스 정보"""
    total_classes: int = Field(..., ge=0, description="총 클래스 수")
    categories: List[DefectCategory] = Field(..., description="카테고리 목록")
    class_hierarchy: Optional[Dict[str, List[str]]] = Field(None, description="클래스 계층 구조")

    class Config:
        json_schema_extra = {
            "example": {
                "total_classes": 13,
                "categories": [
                    {"id": 0, "name": "정상", "supercategory": "품질상태"},
                    {"id": 101, "name": "스크래치", "supercategory": "자동차부품품질"},
                    {"id": 102, "name": "외관 손상", "supercategory": "자동차부품품질"}
                ],
                "class_hierarchy": {
                    "표면_결함": ["스크래치", "외관 손상"],
                    "조립_결함": ["고정 불량", "고정핀 불량", "체결 불량"]
                }
            }
        }


class ModelClassesResponse(BaseDataResponse):
    """모델 클래스 조회 응답 (/model/classes)"""
    data: ModelClassesInfo = Field(..., description="모델 클래스 정보")


class ModelPerformanceResponse(BaseDataResponse):
    """모델 성능 조회 응답 (/model/performance)"""
    data: PerformanceMetrics = Field(..., description="모델 성능 지표")


class ModelStatisticsResponse(BaseDataResponse):
    """모델 통계 조회 응답 (/model/statistics)"""
    data: ModelStatistics = Field(..., description="모델 사용 통계")


class ModelReloadResult(BaseModel):
    """모델 재로딩 결과"""
    success: bool = Field(..., description="재로딩 성공 여부")
    old_version: Optional[str] = Field(None, description="이전 버전")
    new_version: str = Field(..., description="새 버전")
    reload_time_seconds: float = Field(..., ge=0, description="재로딩 소요 시간(초)")
    changes: Optional[List[str]] = Field(None, description="변경 사항")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "old_version": "1.1.0",
                "new_version": "1.2.0",
                "reload_time_seconds": 3.45,
                "changes": [
                    "모델 가중치 업데이트",
                    "새로운 클래스 추가",
                    "성능 최적화"
                ]
            }
        }


class ModelReloadResponse(BaseDataResponse):
    """모델 재로딩 응답 (/model/reload)"""
    data: ModelReloadResult = Field(..., description="모델 재로딩 결과")