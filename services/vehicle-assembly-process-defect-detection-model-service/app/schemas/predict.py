from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union

from app.schemas.common import (
    BaseDataResponse,
    FileInfo,
    ProcessingTime
)


class ImageData(BaseModel):
    """이미지 데이터 모델 (Base64 인코딩)"""
    base64_data: str = Field(..., description="Base64 인코딩된 이미지 데이터")

    class Config:
        json_schema_extra = {
            "example": {
                "base64_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/..."
            }
        }

class PredictRequest(BaseModel):
    """AI 모델 예측 요청 (POST /predict)"""
    image: ImageData = Field(..., description="Base64 인코딩된 이미지 데이터")
    options: Optional[Dict[str, Union[str, int, float, bool]]] = Field(
        default_factory=dict,
        description="예측 옵션"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image": {
                    "base64_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAAcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCABkAGQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD..."
                },
                "options": {
                    "confidence_threshold": 0.8,
                    "return_probabilities": True
                }
            }
        }


class ClassProbability(BaseModel):
    """클래스별 확률"""
    category_id: int = Field(..., description="카테고리 ID")
    category_name: str = Field(..., description="카테고리 이름")
    probability: float = Field(..., ge=0, le=1, description="확률")

    class Config:
        json_schema_extra = {
            "example": {
                "category_id": 101,
                "category_name": "스크래치",
                "probability": 0.85
            }
        }


class PredictionResult(BaseModel):
    """예측 결과 데이터"""
    # 기본 예측 결과
    predicted_category_id: int = Field(..., description="예측된 카테고리 ID")
    predicted_category_name: str = Field(..., description="예측된 카테고리 이름")
    predicted_label: str = Field(..., description="예측된 라벨 (라우터 로그용)")
    confidence: float = Field(..., ge=0, le=1, description="예측 신뢰도")
    is_defective: bool = Field(..., description="불량 여부")

    # 선택적 상세 정보
    class_probabilities: Optional[List[ClassProbability]] = Field(
        None,
        description="모든 카테고리별 확률"
    )

    # 처리 정보
    processing_time: ProcessingTime = Field(..., description="처리 시간 정보")
    model_version: Optional[str] = Field(None, description="사용된 모델 버전")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_category_id": 201,
                "predicted_category_name": "고정 불량",
                "predicted_label": "고정 불량",
                "confidence": 0.89,
                "is_defective": True,
                "class_probabilities": [
                    {"category_id": 0, "category_name": "정상", "probability": 0.08},
                    {"category_id": 201, "category_name": "고정 불량", "probability": 0.89},
                    {"category_id": 203, "category_name": "단차", "probability": 0.03}
                ],
                "processing_time": {
                    "total_seconds": 0.145,
                    "preprocessing_seconds": 0.035,
                    "inference_seconds": 0.095,
                    "postprocessing_seconds": 0.015
                },
                "model_version": "1.2.0"
            }
        }


class PredictResponse(BaseDataResponse):
    """AI 모델 예측 응답 (POST /predict)"""
    data: PredictionResult = Field(..., description="예측 결과")


class FilePredictionResult(PredictionResult):
    """파일 예측 결과 (파일 정보 포함)"""
    file_info: FileInfo = Field(..., description="업로드된 파일 정보")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_category_id": 101,
                "predicted_category_name": "스크래치",
                "predicted_label": "스크래치",
                "confidence": 0.85,
                "is_defective": True,
                "class_probabilities": [
                    {"category_id": 0, "category_name": "정상", "probability": 0.12},
                    {"category_id": 101, "category_name": "스크래치", "probability": 0.85},
                    {"category_id": 201, "category_name": "고정 불량", "probability": 0.03}
                ],
                "processing_time": {
                    "total_seconds": 0.125,
                    "preprocessing_seconds": 0.030,
                    "inference_seconds": 0.080,
                    "postprocessing_seconds": 0.015
                },
                "file_info": {
                    "filename": "202_201_10_sample.jpg",
                    "content_type": "image/jpeg",
                    "size": 1024000,
                    "width": 4032,
                    "height": 1908
                },
                "model_version": "1.2.0"
            }
        }


class FilePredictResponse(BaseDataResponse):
    """파일 업로드를 통한 예측 응답 (POST /predict/file)"""
    data: FilePredictionResult = Field(..., description="파일 예측 결과")


class BatchPredictionItem(BaseModel):
    """배치 예측 항목"""
    filename: str = Field(..., description="파일명")
    success: bool = Field(..., description="예측 성공 여부")
    result: Optional[FilePredictionResult] = Field(None, description="예측 결과")
    error: Optional[str] = Field(None, description="오류 메시지")

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "car_part_001.jpg",
                "success": True,
                "result": {
                    "predicted_category_id": 0,
                    "predicted_category_name": "정상",
                    "predicted_label": "정상",
                    "confidence": 0.95,
                    "is_defective": False,
                    "processing_time": {"total_seconds": 0.120},
                    "file_info": {
                        "filename": "car_part_001.jpg",
                        "content_type": "image/jpeg",
                        "size": 1500000,
                        "width": 3840,
                        "height": 2160
                    },
                    "model_version": "1.2.0"
                },
                "error": None
            }
        }


class BatchSummary(BaseModel):
    """배치 처리 요약"""
    total_count: int = Field(..., ge=0, description="총 파일 수")
    successful_count: int = Field(..., ge=0, description="성공한 파일 수")
    failed_count: int = Field(..., ge=0, description="실패한 파일 수")
    defective_count: int = Field(..., ge=0, description="불량으로 판정된 파일 수")
    normal_count: int = Field(..., ge=0, description="정상으로 판정된 파일 수")
    average_confidence: Optional[float] = Field(None, ge=0, le=1, description="평균 신뢰도")
    total_processing_time: float = Field(..., ge=0, description="총 처리 시간(초)")

    class Config:
        json_schema_extra = {
            "example": {
                "total_count": 5,
                "successful_count": 4,
                "failed_count": 1,
                "defective_count": 2,
                "normal_count": 2,
                "average_confidence": 0.87,
                "total_processing_time": 0.650
            }
        }


class BatchPredictionData(BaseModel):
    """배치 예측 데이터"""
    results: List[BatchPredictionItem] = Field(..., description="개별 예측 결과들")
    summary: BatchSummary = Field(..., description="배치 처리 요약")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "filename": "part_001.jpg",
                        "success": True,
                        "result": {
                            "predicted_category_id": 0,
                            "predicted_category_name": "정상",
                            "predicted_label": "정상",
                            "confidence": 0.95,
                            "is_defective": False,
                            "processing_time": {"total_seconds": 0.120},
                            "file_info": {
                                "filename": "part_001.jpg",
                                "content_type": "image/jpeg",
                                "size": 1500000
                            },
                            "model_version": "1.2.0"
                        },
                        "error": None
                    },
                    {
                        "filename": "part_002.jpg",
                        "success": True,
                        "result": {
                            "predicted_category_id": 101,
                            "predicted_category_name": "스크래치",
                            "predicted_label": "스크래치",
                            "confidence": 0.82,
                            "is_defective": True,
                            "processing_time": {"total_seconds": 0.130},
                            "file_info": {
                                "filename": "part_002.jpg",
                                "content_type": "image/jpeg",
                                "size": 1200000
                            },
                            "model_version": "1.2.0"
                        },
                        "error": None
                    }
                ],
                "summary": {
                    "total_count": 2,
                    "successful_count": 2,
                    "failed_count": 0,
                    "defective_count": 1,
                    "normal_count": 1,
                    "average_confidence": 0.885,
                    "total_processing_time": 0.250
                }
            }
        }


class BatchPredictResponse(BaseDataResponse):
    """배치 파일 예측 응답 (POST /predict/batch)"""
    data: BatchPredictionData = Field(..., description="배치 예측 데이터")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "배치 예측이 완료되었습니다",
                "timestamp": "2025-08-06T22:47:02Z",
                "data": {
                    "results": [
                        {
                            "filename": "part_001.jpg",
                            "success": True,
                            "result": {
                                "predicted_category_id": 0,
                                "predicted_category_name": "정상",
                                "predicted_label": "정상",
                                "confidence": 0.95,
                                "is_defective": False
                            }
                        }
                    ],
                    "summary": {
                        "total_count": 3,
                        "successful_count": 2,
                        "failed_count": 1,
                        "defective_count": 1,
                        "normal_count": 1,
                        "average_confidence": 0.87,
                        "total_processing_time": 0.650
                    }
                }
            }
        }