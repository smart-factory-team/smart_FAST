from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from datetime import datetime
from enum import Enum


class BaseResponse(BaseModel):
    """기본 응답 모델"""
    success: bool = Field(..., description="요청 성공 여부")
    message: str = Field(..., description="응답 메시지")
    timestamp: datetime = Field(default_factory=datetime.now, description="응답 시간")
    request_id: Optional[str] = Field(None, description="요청 ID")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BaseDataResponse(BaseResponse):
    """데이터를 포함하는 기본 응답 모델"""
    data: Any = Field(..., description="응답 데이터")


class FileInfo(BaseModel):
    """파일 정보 모델"""
    filename: Optional[str] = Field(None, description="파일명")
    content_type: Optional[str] = Field(None, description="파일 타입")
    size: Optional[int] = Field(None, ge=0, description="파일 크기(bytes)")
    width: Optional[int] = Field(None, ge=1, description="이미지 너비")
    height: Optional[int] = Field(None, ge=1, description="이미지 높이")

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "202_201_10_sample.jpg",
                "content_type": "image/jpeg",
                "size": 1024000,
                "width": 4032,
                "height": 1908
            }
        }


class ProcessingTime(BaseModel):
    """처리 시간 정보"""
    total_seconds: float = Field(..., ge=0, description="총 처리 시간(초)")
    preprocessing_seconds: Optional[float] = Field(None, ge=0, description="전처리 시간(초)")
    inference_seconds: Optional[float] = Field(None, ge=0, description="추론 시간(초)")
    postprocessing_seconds: Optional[float] = Field(None, ge=0, description="후처리 시간(초)")

    class Config:
        json_schema_extra = {
            "example": {
                "total_seconds": 0.125,
                "preprocessing_seconds": 0.025,
                "inference_seconds": 0.085,
                "postprocessing_seconds": 0.015
            }
        }


class HealthStatus(str, Enum):
    """헬스 상태 열거형"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class ModelStatus(str, Enum):
    """모델 상태 열거형"""
    LOADED = "loaded"
    LOADING = "loading"
    NOT_LOADED = "not_loaded"
    ERROR = "error"

class ProcessType(str, Enum):
    ASSEMBLY = "조립"

class WorkType(str, Enum):
    """작업 분류 열거형"""
    ASSEMBLY = "조립"      # 조립 작업
    PAINTING = "도장"      # 도장 작업


class PartType(str, Enum):
    """부품 분류 열거형"""
    DOOR = "도어"
    GRILL = "라디에이터 그릴"
    ROOFSD = "루프사이드"
    WIRING = "배선"
    BUMPER = "범퍼"
    COWLCVR = "카울커버"
    CONN = "커넥터"
    TAILLMP = "테일 램프"
    FRAME = "프레임"
    HEADLMP = "헤드램프"
    FENDER = "휀더"

class DefectType(str, Enum):
    """불량 타입 열거형"""
    NORMAL = "정상"
    SCRATCH = "스크래치"
    EXTDMG = "외관 손상"
    FIXERR = "고정 불량"
    PINERR = "고정핀 불량"
    GAP = "단차"
    SEALERR = "실링 불량"
    JOINTERR = "연계 불량"
    PLAY = "유격 불량"
    MOUNTERR = "장착 불량"
    CONNECT = "체결 불량"
    HEMERR = "헤밍 불량"
    HOLEDEF = "홀 변형"


class DefectCategory(BaseModel):
    """불량 카테고리 정보"""
    id: int = Field(..., description="카테고리 ID")
    name: str = Field(..., description="불량 이름")
    supercategory: str = Field(..., description="상위 카테고리")
    description: Optional[str] = Field(None, description="불량 설명")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 101,
                "name": "스크래치",
                "supercategory": "자동차부품품질",
                "description": "부품 표면의 긁힘 현상"
            }
        }


class QualityType(str, Enum):
    """품질 분류 열거형"""
    GOOD = "양품"          # 양품 (정상)
    DEFECT = "불량"        # 불량


class DefectAttributes(BaseModel):
    """불량 속성 정보"""
    work: WorkType = Field(..., description="작업 분류")
    part: PartType = Field(..., description="부품 분류")
    quality: QualityType = Field(..., description="품질 분류")

    class Config:
        json_schema_extra = {
            "example": {
                "work": "조립",
                "part": "배선",
                "quality": "양품"
            }
        }

class IssueCode(BaseModel):
    """이슈 코드 생성 모델 (공정-부품-불량타입)"""
    process_type: ProcessType = Field(..., description="공정 분류")
    part_type: PartType = Field(..., description="부품 분류")
    defect_type: DefectType = Field(..., description="불량 타입")
    issue_code: str = Field(..., description="생성된 이슈 코드")
    description: str = Field(..., description="이슈 설명")
    generated_at: datetime = Field(..., description="생성 시간")

    class Config:
        json_schema_extra = {
            "example": {
                "part_type": "배선",
                "defect_type": "고정 불량",
                "issue_code": "ASBP-WIRING-FIXERR-240101120000",
                "description": "조립 공정에서 배선 부품의 고정 불량 발생",
                "generated_at": "2024-01-01T12:00:00Z"
            }
        }


class BoundingBox(BaseModel):
    """바운딩 박스 좌표 (COCO 형식)"""
    x: float = Field(..., ge=0, description="좌상단 X 좌표")
    y: float = Field(..., ge=0, description="좌상단 Y 좌표")
    width: float = Field(..., gt=0, description="박스 너비")
    height: float = Field(..., gt=0, description="박스 높이")

    class Config:
        json_schema_extra = {
            "example": {
                "x": 1364.06,
                "y": 816.38,
                "width": 956.94,
                "height": 840.90
            }
        }


# 실제 데이터셋의 카테고리 정의
DEFECT_CATEGORIES = [
    {"id": 0, "name": "고정 불량_불량품", "supercategory": "조립_결함"},
    {"id": 1, "name": "고정 불량_양품", "supercategory": "조립_결함"},
    {"id": 2, "name": "고정핀 불량_불량품", "supercategory": "조립_결함"},
    {"id": 3, "name": "고정핀 불량_양품", "supercategory": "조립_결함"},
    {"id": 4, "name": "단차_불량품", "supercategory": "조립_결함"},
    {"id": 5, "name": "단차_양품", "supercategory": "조립_결함"},
    {"id": 6, "name": "스크래치_불량품", "supercategory": "도장_결함"},
    {"id": 7, "name": "스크래치_양품", "supercategory": "도장_결함"},
    {"id": 8, "name": "실링 불량_불량품", "supercategory": "조립_결함"},
    {"id": 9, "name": "실링 불량_양품", "supercategory": "조립_결함"},
    {"id": 10, "name": "연계 불량_불량품", "supercategory": "조립_결함"},
    {"id": 11, "name": "연계 불량_양품", "supercategory": "조립_결함"},
    {"id": 12, "name": "외관 손상_불량품", "supercategory": "도장_결함"},
    {"id": 13, "name": "외관 손상_양품", "supercategory": "도장_결함"},
    {"id": 14, "name": "유격 불량_불량품", "supercategory": "조립_결함"},
    {"id": 15, "name": "유격 불량_양품", "supercategory": "조립_결함"},
    {"id": 16, "name": "장착 불량_불량품", "supercategory": "조립_결함"},
    {"id": 17, "name": "장착 불량_양품", "supercategory": "조립_결함"},
    {"id": 18, "name": "체결 불량_불량품", "supercategory": "조립_결함"},
    {"id": 19, "name": "체결 불량_양품", "supercategory": "조립_결함"},
    {"id": 20, "name": "헤밍 불량_불량품", "supercategory": "조립_결함"},
    {"id": 21, "name": "헤밍 불량_양품", "supercategory": "조립_결함"},
    {"id": 22, "name": "홀 변형_불량품", "supercategory": "조립_결함"},
    {"id": 23, "name": "홀 변형_양품", "supercategory": "조립_결함"}
]

# 이슈 코드 매핑 테이블
PROCESS_CODE_FIXED = "ASBP"  # 고정된 공정 코드

PART_CODE_MAP = {
    PartType.DOOR: "DOOR",
    PartType.GRILL: "GRILL",
    PartType.ROOFSD: "ROOFSD",
    PartType.WIRING: "WIRING",
    PartType.BUMPER: "BUMPER",
    PartType.COWLCVR: "COWLCVR",
    PartType.CONN: "CONN",
    PartType.TAILLMP: "TAILLMP",
    PartType.FRAME: "FRAME",
    PartType.HEADLMP: "HEADLMP",
    PartType.FENDER: "FENDER"
}


DEFECT_CODE_MAP = {
    DefectType.NORMAL: "NORMAL",
    DefectType.SCRATCH: "SCRATCH",
    DefectType.EXTDMG: "EXTDMG",
    DefectType.FIXERR: "FIXERR",
    DefectType.PINERR: "PINERR",
    DefectType.GAP: "GAP",
    DefectType.SEALERR: "SEALERR",
    DefectType.JOINTERR: "JOINTERR",
    DefectType.PLAY: "PLAY",
    DefectType.MOUNTERR: "MOUNTERR",
    DefectType.CONNECT: "CONNECT",
    DefectType.HEMERR: "HEMERR",
    DefectType.HOLEDEF: "HOLEDEF"
}

def get_defect_category_by_id(category_id: int) -> Optional[DefectCategory]:
    """ID로 불량 카테고리 조회"""
    for cat in DEFECT_CATEGORIES:
        if cat["id"] == category_id:
            return DefectCategory(**cat)
    return None


def is_defective_category(category_id: int) -> bool:
    """불량 카테고리인지 확인 (0: 정상, 나머지: 불량)"""
    return category_id != 0

def generate_issue_code(
    part_type: PartType,
    defect_type: DefectType,
    timestamp: Optional[datetime] = None
) -> IssueCode:
    """이슈 코드 생성 함수

    형식: ASBP-{부품코드}-{불량코드}-{YYMMDDHHMMSS}
    예시: ASBP-WIRING-FIXERR-240101120000 (고정공정-배선-고정불량-2024년1월1일12시00분00초)
    """
    process_code = PROCESS_CODE_FIXED
    part_code = PART_CODE_MAP.get(part_type, "UNKNOWN")
    defect_code = DEFECT_CODE_MAP.get(defect_type, "UNKNOWN")

    # 시간을 YYMMDDHHMMSS 형식으로 변환
    time_code = timestamp.strftime("%y%m%d%H%M%S")

    issue_code = f"{process_code}-{part_code}-{defect_code}-{time_code}"

    # ProcessType.ASSEMBLY를 기본값으로 사용 (ASBP가 고정이므로)
    process_type = ProcessType.ASSEMBLY

    description = f"{process_type.value} 공정에서 {part_type.value} 부품의 {defect_type.value}"
    if defect_type != DefectType.NORMAL:
        description += " 발생"
    else:
        description += " - 정상 상태"

    return IssueCode(
        process_type=process_type,
        part_type=part_type,
        defect_type=defect_type,
        issue_code=issue_code,
        description=description,
        generated_at=timestamp
    )


def parse_issue_code(issue_code: str) -> Optional[Dict[str, str]]:
    """이슈 코드 파싱 함수

    Args:
        issue_code: 파싱할 이슈 코드 (예: ASBP-WIRING-FIXERR-240101120000)

    Returns:
        파싱된 정보 딕셔너리 또는 None
    """
    try:
        parts = issue_code.split('-')
        if len(parts) != 4:
            return None

        process_code, part_code, defect_code, time_code = parts

        # 공정 코드는 ASBP 고정이므로 검증
        if process_code != PROCESS_CODE_FIXED:
            return None

        # 시간 코드 파싱 (YYMMDDHHMMSS)
        if len(time_code) != 12:
            return None

        try:
            # 시간 문자열을 datetime 객체로 변환
            parsed_time = datetime.strptime(time_code, "%y%m%d%H%M%S")
        except ValueError:
            return None

        # 역매핑을 위한 딕셔너리 생성
        reverse_part_map = {v: k for k, v in PART_CODE_MAP.items()}
        reverse_defect_map = {v: k for k, v in DEFECT_CODE_MAP.items()}

        return {
            "process_type": ProcessType.ASSEMBLY,  # ASBP는 항상 조립
            "part_type": reverse_part_map.get(part_code),
            "defect_type": reverse_defect_map.get(defect_code),
            "generated_at": parsed_time,
            "time_code": time_code
        }
    except (ValueError, IndexError):
        return None

def get_timestamp_from_issue_code(issue_code: str) -> Optional[datetime]:
   """이슈 코드에서 생성 시간만 추출하는 헬퍼 함수"""
   parsed = parse_issue_code(issue_code)
   return parsed.get("generated_at") if parsed else None