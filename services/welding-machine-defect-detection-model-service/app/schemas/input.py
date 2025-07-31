from pydantic import BaseModel, Field, field_validator
from typing import List, Literal


class SensorData(BaseModel):
    signal_type: Literal["cur", "vib"] = Field(
        description="Type of sensor signal: 'cur' for current, 'vib' for vibration"
    )
    values: List[float] = Field(
        min_items=1,
        max_items=10000,
        description="List of sensor readings as float values"
    )

    @field_validator('values')
    @classmethod
    def validate_values(cls, v):
        if any(val < 0 for val in v):
            raise ValueError('Sensor values must be non-negative')
        return v
