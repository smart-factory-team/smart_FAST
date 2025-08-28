from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List

SEQUENCE_LENGTH = 20


class SensorData(BaseModel):
    AI0_Vibration: List[float] = Field(
        ..., description="upper vibration time series data list"
    )
    AI1_Vibration: List[float] = Field(
        ..., description="lower vibration time series data list"
    )
    AI2_Current: List[float] = Field(..., description="current time series data list")

    @field_validator("AI0_Vibration", "AI1_Vibration", "AI2_Current")
    @classmethod
    def check_min_length(cls, v, info):
        if len(v) < SEQUENCE_LENGTH:
            raise ValueError(
                f"{info.field_name} 필드는 최소 {SEQUENCE_LENGTH}개의 데이터 포인트를 가져야 합니다."
            )
        return v

    @model_validator(mode="after")
    def check_list_lengths_are_equal(self) -> "SensorData":
        len_ai0 = len(self.AI0_Vibration)
        len_ai1 = len(self.AI1_Vibration)
        len_ai2 = len(self.AI2_Current)

        if not (len_ai0 == len_ai1 == len_ai2):
            raise ValueError("모든 센서 데이터 리스트의 길이는 동일해야 합니다.")
        return self
