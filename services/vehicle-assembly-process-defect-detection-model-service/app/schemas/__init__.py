"""
Pydantic 스키마 모듈 - 필요한 엔드포인트만 지원

지원하는 엔드포인트:
- GET / - 서비스 정보
- GET /health - 헬스 체크
- GET /ready - AI 모델 로딩 상태 체크
- GET /startup - 서비스 시작 준비 체크
- POST /predict - AI 모델 예측
- POST /predict/file - 파일 업로드를 통한 예측
- GET /model/info - 모델 정보 조회
"""

# === 공통 스키마 ===
from .common import (
    # 기본 응답 모델들
    BaseResponse,
    BaseDataResponse,

    # 공통 데이터 모델들
    FileInfo,
    ProcessingTime,
    DefectCategory,
    IssueCode,

    # 열거형들
    HealthStatus,
    ModelStatus,
    ProcessType,
    PartType,
    DefectType,

    # 상수 및 헬퍼 함수들
    DEFECT_CATEGORIES,
    get_defect_category_by_id,
    is_defective_category,
    generate_issue_code,
    parse_issue_code
)

# === 서비스 정보 스키마 ===
from .root import (
    ServiceInfo,
    ServiceInfoResponse,
    ServiceEndpoints,
    ApiDocs
)

# === 헬스체크 스키마 ===
from .health import (
    # 헬스체크 데이터 모델들
    HealthData,
    ModelInfo,
    ReadinessData,
    StartupData,
    StartupCheck,

    # 헬스체크 응답 모델들
    HealthResponse,
    ReadinessResponse,
    StartupResponse
)

# === 예측 스키마 ===
from .predict import (
    # 요청 모델들
    ImageData,
    PredictRequest,

    # 결과 데이터 모델들
    ClassProbability,
    PredictionResult,
    FilePredictionResult,
    BatchPredictionItem,
    BatchSummary,
    BatchPredictionData,

    # 응답 모델들
    PredictResponse,
    FilePredictResponse,
    BatchPredictResponse
)

# === 모델 정보 스키마 ===
from .model import (
    # 모델 정보 모델들
    ModelConfiguration,
    ModelStatistics,
    ModelVersionInfo,
    ModelInfo,
    PerformanceMetrics,
    ModelClassesInfo,
    ModelReloadResult,

    # 응답 모델들 - 이것들을 추가해야 함!
    ModelInfoResponse,
    ModelClassesResponse,
    ModelPerformanceResponse,
    ModelStatisticsResponse,
    ModelReloadResponse
)

# === 개발 편의용 별칭 ===
HealthResp = HealthResponse
PredictResp = PredictResponse
FilePredictResp = FilePredictResponse
ModelInfoResp = ModelInfoResponse
ServiceInfoResp = ServiceInfoResponse

# === 헬퍼 함수들 ===
def create_error_response(
    message: str,
    error_code: str = None,
    details: dict = None,
    request_id: str = None
) -> dict:
    """에러 응답 생성 헬퍼"""
    from datetime import datetime

    return {
        "success": False,
        "message": message,
        "error_code": error_code or "GENERAL_ERROR",
        "details": details or {},
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id
    }


def create_success_response(
    message: str,
    data: any = None,
    request_id: str = None
) -> dict:
    """성공 응답 생성 헬퍼"""
    from datetime import datetime

    response = {
        "success": True,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id
    }

    if data is not None:
        response["data"] = data

    return response


# === 모듈 정보 ===
__version__ = "1.0.0"
__description__ = "자동차 의장 불량 탐지 API 스키마"

# === 익스포트할 주요 클래스들 ===
__all__ = [
    # 공통
    "BaseResponse",
    "BaseDataResponse",
    "FileInfo",
    "ProcessingTime",
    "DefectCategory",
    "IssueCode",
    "ProcessType",
    "PartType",
    "DefectType",

    # 서비스 정보
    "ServiceInfoResponse",
    "ServiceInfo",

    # 헬스체크
    "HealthResponse",
    "ReadinessResponse",
    "StartupResponse",
    "HealthData",
    "ReadinessData",
    "StartupData",

    # 예측
    "PredictRequest",
    "PredictResponse",
    "FilePredictResponse",
    "BatchPredictResponse",
    "PredictionResult",
    "FilePredictionResult",
    "ImageData",
    "ClassProbability",

    # 모델
    "ModelInfoResponse",
    "ModelClassesResponse",
    "ModelPerformanceResponse",
    "ModelStatisticsResponse",
    "ModelReloadResponse",
    "ModelInfo",
    "ModelConfiguration",
    "ModelStatistics",
    "PerformanceMetrics",

    # 헬퍼 함수들
    "create_error_response",
    "create_success_response",
    "get_defect_category_by_id",
    "is_defective_category",
    "generate_issue_code",
    "parse_issue_code",
    "get_timestamp_from_issue_code",

    # 별칭들
    "HealthResp",
    "PredictResp",
    "FilePredictResp",
    "ModelInfoResp",
    "ServiceInfoResp"
]