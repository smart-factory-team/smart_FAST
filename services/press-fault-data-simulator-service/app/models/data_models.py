from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PredictionRequest(BaseModel):
    """
    고장 예측 API(/predict)로 전송할 요청 본문의 데이터 모델.
    이 시뮬레이터가 생성해야 할 데이터의 최종 형태.
    """

    AI0_Vibration: List[float] = Field(..., description="시계열 진동 데이터 리스트")
    AI1_Vibration: List[float] = Field(..., description="진동 데이터 리스트")
    AI2_Current: List[float] = Field(..., description="전류 데이터 리스트")


class PredictionResult(BaseModel):
    """
    고장 예측 API(/predict)로부터 수신할 응답 결과의 데이터 모델.
    이 시뮬레이터가 수신하고 처리해야 할 데이터의 형태.
    """

    prediction: str
    reconstruction_error: float
    is_fault: bool
    fault_probability: float
    attribute_errors: Optional[Dict[str, float]] = None


class SimulatorStatus(BaseModel):
    """
    시뮬레이터의 현재 상태를 나타내는 데이터 모델
    """

    status: str = Field(description="시뮬레이터의 현재 상태")
    job_id: Optional[str] = Field(
        default=None, description="현재 실행 중인 스케줄러 작업의 ID"
    )
    next_run_time: Optional[str] = Field(
        default=None, description="다음 작업 실행 예정 시간 (ISO 8601 형식)"
    )
    last_run_result: Optional[str] = Field(
        default=None, description="마지막으로 실행된 작업의 성공/실패 여부"
    )
