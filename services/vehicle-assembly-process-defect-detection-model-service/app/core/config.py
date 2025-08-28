"""
애플리케이션 설정 모듈

환경변수와 기본값을 통해 애플리케이션의 모든 설정을 관리합니다.
Pydantic Settings를 사용하여 타입 안정성과 검증을 제공합니다.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""

    # === 기본 애플리케이션 설정 ===
    PROJECT_NAME: str = "자동차 의장 공정 불량 탐지 API"
    PROJECT_DESCRIPTION: str = "자동차 의장 공정에서 발생하는 부품 불량을 탐지하는 이미지 분류 API 서비스"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="production", description="실행 환경")
    DEBUG: bool = Field(default=False, description="디버그 모드")

    # === 서버 설정 ===
    HOST: str = Field(default="0.0.0.0", description="서버 호스트")
    PORT: int = Field(default=8005, description="서버 포트")
    API_V1_STR: str = "/api/v1"

    # === AI 모델 설정 ===
    # HuggingFace 모델 정보
    MODEL_NAME: str = Field(
        default="23smartfactory/vehicle-assembly-process-defect-detection-model",
        description="HuggingFace 모델 이름"
    )
    MODEL_VERSION: str = Field(
        default="v1.0",
        description="모델 버전"
    )
    MODEL_PATH: str = Field(
        default="23smartfactory/vehicle-assembly-process-defect-detection-model",
        description="HuggingFace 모델 경로 또는 로컬 경로"
    )

    # GPU/CPU 설정
    USE_GPU: bool = Field(default=False, description="GPU 사용 여부")
    GPU_MEMORY_LIMIT: Optional[int] = Field(default=None, description="GPU 메모리 제한")

    # 모델 캐시 설정
    MODEL_CACHE_DIR: str = Field(
        default="./model_cache",
        description="모델 캐시 디렉토리"
    )

    # === 파일 업로드 설정 ===
    # 지원하는 이미지 타입
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/bmp",
        "image/tiff"
    ]

    # 파일 크기 제한 (bytes)
    MAX_IMAGE_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="최대 이미지 크기 (bytes)"
    )

    # 배치 처리 제한
    BATCH_SIZE_LIMIT: int = Field(default=10, description="배치 처리 제한")

    # === 디렉토리 설정 ===
    # 기본 디렉토리들
    MODEL_BASE_PATH: str = Field(default="./models", description="모델 기본 경로")
    UPLOAD_DIR: str = Field(default="./uploads", description="업로드 디렉토리")
    TEMP_DIR: str = Field(default="./temp", description="임시 디렉토리")
    LOG_DIR: str = Field(default="./logs", description="로그 디렉토리")

    # === 로깅 설정 ===
    LOG_LEVEL: str = Field(default="INFO", description="로그 레벨")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_SIZE: int = Field(default=10 * 1024 * 1024, description="로그 파일 최대 크기")  # 10MB
    LOG_FILE_BACKUP_COUNT: int = Field(default=5, description="로그 파일 백업 개수")

    # === 보안 설정 ===
    # API 키 인증 (선택적)
    REQUIRE_API_KEY: bool = Field(default=False, description="API 키 인증 필요 여부")
    API_KEY: Optional[str] = Field(default=None, description="API 키")
    ADMIN_TOKEN: Optional[str] = Field(default=None, description="관리자 토큰")

    # CORS 설정
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="허용할 CORS 오리진들 (콤마로 구분)"
    )

    # === 성능 및 모니터링 설정 ===
    # 요청 타임아웃 (초)
    REQUEST_TIMEOUT: int = Field(default=30, description="요청 타임아웃 (초)")

    # 모델 예측 타임아웃 (초)
    PREDICTION_TIMEOUT: int = Field(default=10, description="모델 예측 타임아웃 (초)")

    # 메트릭 수집 여부
    ENABLE_METRICS: bool = Field(default=True, description="메트릭 수집 여부")

    # === 개발/테스트 설정 ===
    # 테스트 모드 (실제 모델 로딩 생략)
    TEST_MODE: bool = Field(default=False, description="테스트 모드")

    # 더미 예측 사용 (개발용)
    USE_DUMMY_PREDICTIONS: bool = Field(default=False, description="더미 예측 사용")

    # === HuggingFace 관련 설정 ===
    # HuggingFace 토큰 (private 모델 사용시)
    HF_TOKEN: Optional[str] = Field(default=None, description="HuggingFace 토큰")

    # 오프라인 모드 (인터넷 연결 없이 캐시된 모델만 사용)
    HF_OFFLINE: bool = Field(default=False, description="HuggingFace 오프라인 모드")

    # === 특화 모델 설정 (의장공정용) ===
    # 기본 신뢰도 임계값
    DEFAULT_CONFIDENCE_THRESHOLD: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="기본 신뢰도 임계값"
    )

    # 불량 판정 임계값
    DEFECT_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="불량 판정 임계값"
    )

    # 클래스별 확률 반환 여부
    RETURN_CLASS_PROBABILITIES: bool = Field(
        default=True,
        description="클래스별 확률 반환 여부"
    )

    @field_validator("CORS_ORIGINS", mode="before")
    def parse_cors_origins(cls, v):
        """CORS origins를 문자열에서 리스트로 변환"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """로그 레벨 검증"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("MODEL_PATH")
    def validate_model_path(cls, v):
        """모델 경로 검증"""
        if not v:
            raise ValueError("MODEL_PATH cannot be empty")
        return v

    def create_directories(self):
        """필요한 디렉토리들 생성 (예외 처리 포함)"""
        directories = [
            self.MODEL_BASE_PATH,
            self.UPLOAD_DIR,
            self.TEMP_DIR,
            self.LOG_DIR,
            self.MODEL_CACHE_DIR
        ]

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Directory created/verified: {directory}")
            except PermissionError as e:
                print(f"Warning: Cannot create directory {directory}: {e}")
                continue
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
                continue

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True
    }


# === 사전 정의된 모델 설정들 ===

class ModelConfigs:
    """사전 정의된 모델 설정들"""

    # 예시: ResNet 기반 모델들
    RESNET50_BASE = {
        "MODEL_NAME": "microsoft/resnet-50",
        "MODEL_PATH": "microsoft/resnet-50",
        "MODEL_VERSION": "1.0.0"
    }

    # 의장공정 특화 모델
    DEFECT_DETECTION_V1 = {
        "MODEL_NAME": "23smartfactory/vehicle-assembly-process-defect-detection-model",
        "MODEL_PATH": "23smartfactory/vehicle-assembly-process-defect-detection-model",
        "MODEL_VERSION": "1.0.0"
    }

    # 로컬 모델 (파인튜닝된 모델)
    LOCAL_FINETUNED = {
        "MODEL_NAME": "local-car-defect-detector",
        "MODEL_PATH": "./models/finetuned-resnet50-car-defects",
        "MODEL_VERSION": "1.0.0"
    }


def get_model_config(model_type: str = "defect_v1") -> dict:
    """
    모델 타입에 따른 설정 반환

    Args:
        model_type: 모델 타입 ("resnet50", "vit", "defect_v1", "defect_v2", "local")

    Returns:
        dict: 모델 설정 딕셔너리
    """
    configs = {
        "resnet50": ModelConfigs.RESNET50_BASE,
        "defect_v1": ModelConfigs.DEFECT_DETECTION_V1,
        "local": ModelConfigs.LOCAL_FINETUNED
    }

    return configs.get(model_type, ModelConfigs.DEFECT_DETECTION_V1)


# === 환경별 설정 ===

class DevelopmentSettings(Settings):
    """개발 환경 설정"""
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    USE_DUMMY_PREDICTIONS: bool = True


class ProductionSettings(Settings):
    """프로덕션 환경 설정"""
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    REQUIRE_API_KEY: bool = False  # 필요시 True로 변경
    USE_DUMMY_PREDICTIONS: bool = False


class TestSettings(Settings):
    """테스트 환경 설정"""
    ENVIRONMENT: str = "test"
    DEBUG: bool = True
    TEST_MODE: bool = True
    USE_DUMMY_PREDICTIONS: bool = True
    LOG_LEVEL: str = "WARNING"


def get_settings() -> Settings:
    """
    환경에 따른 설정 인스턴스 반환

    Returns:
        Settings: 환경별 설정 인스턴스
    """
    env = os.getenv("ENVIRONMENT", "production").lower()

    if env == "production":
        settings_instance = ProductionSettings()
    elif env == "test":
        settings_instance = TestSettings()
    else:
        settings_instance = DevelopmentSettings()

    # 필요한 디렉토리 생성
    settings_instance.create_directories()

    return settings_instance


# 전역 설정 인스턴스
settings = get_settings()


# === 설정 검증 및 출력 ===

def print_current_config():
    """현재 설정 출력 (디버깅용)"""
    print("=== 현재 애플리케이션 설정 ===")
    print(f"환경: {settings.ENVIRONMENT}")
    print(f"디버그 모드: {settings.DEBUG}")
    print(f"포트: {settings.PORT}")
    print(f"모델: {settings.MODEL_NAME}")
    print(f"모델 경로: {settings.MODEL_PATH}")
    print(f"GPU 사용: {settings.USE_GPU}")
    print(f"로그 레벨: {settings.LOG_LEVEL}")
    print("=" * 30)


if __name__ == "__main__":
    # 설정 테스트
    print_current_config()

    # 모델 설정 예시 출력
    print("\n=== 사용 가능한 모델 설정들 ===")
    for model_type in ["resnet50", "vit", "defect_v1", "local"]:
        config = get_model_config(model_type)
        print(f"{model_type}: {config['MODEL_NAME']}")