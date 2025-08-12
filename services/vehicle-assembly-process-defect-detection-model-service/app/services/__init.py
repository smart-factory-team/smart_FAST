"""
Services 패키지

의장공정 부품 불량 탐지 서비스의 핵심 비즈니스 로직을 담당하는 서비스들입니다.

서비스 구성:
- ModelManager: AI 모델 로딩 및 관리
- PredictionService: 이미지 불량 예측
- ModelService: 모델 정보 및 성능 관리
"""

from .model_manager import DefectDetectionModelManager
from .prediction_service import PredictionService
from .model_service import ModelService

# 버전 정보
__version__ = "1.0.0"
__description__ = "자동차 의장 공정 불량 탐지 서비스"

# 주요 서비스 클래스들
__all__ = [
    "DefectDetectionModelManager",
    "PredictionService",
    "ModelService"
]

# 서비스 정보
SERVICE_INFO = {
    "model_manager": {
        "class": DefectDetectionModelManager,
        "description": "AI 모델 로딩, 관리 및 예측 수행",
        "features": [
            "HuggingFace 모델 지원",
            "GPU/CPU 자동 감지",
            "비동기 모델 로딩",
            "성능 통계 관리"
        ]
    },
    "prediction_service": {
        "class": PredictionService,
        "description": "다양한 형태의 이미지 예측 서비스",
        "features": [
            "JSON 데이터 예측",
            "파일 업로드 예측",
            "배치 예측",
            "이미지 검증 및 전처리"
        ]
    },
    "model_service": {
        "class": ModelService,
        "description": "모델 정보 조회 및 관리",
        "features": [
            "모델 정보 조회",
            "성능 지표 관리",
            "모델 재로딩",
            "상태 모니터링"
        ]
    }
}


def get_service_info() -> dict:
    """
    서비스 정보 반환

    Returns:
        dict: 서비스 정보
    """
    return SERVICE_INFO


def validate_services() -> dict:
    """
    서비스 유효성 검증

    Returns:
        dict: 검증 결과
    """
    validation_results = {}

    for service_name, service_info in SERVICE_INFO.items():
        try:
            service_class = service_info["class"]
            validation_results[service_name] = {
                "available": True,
                "class_name": service_class.__name__,
                "module": service_class.__module__
            }
        except Exception as e:
            validation_results[service_name] = {
                "available": False,
                "error": str(e)
            }

    return validation_results