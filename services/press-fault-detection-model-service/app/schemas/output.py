from pydantic import BaseModel, Field
from typing import Dict, Optional


class PredictionResponse(BaseModel):
    prediction: str
    reconstruction_error: Optional[float]
    is_fault: bool
    fault_probabilith: float = Field(
        ...,
        description="전체 시퀀스 중 고장으로 판정된 시퀀스의 비율 (0.0 ~ 1.0)",
        ge=0,  # Greater than or equal to 0
        le=1,  # Less than or equal to 1
    )
    attribute_errors: Optional[Dict[str, Optional[float]]] = None
