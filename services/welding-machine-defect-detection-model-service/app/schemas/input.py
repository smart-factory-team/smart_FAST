from pydantic import BaseModel
from typing import List, Literal


class SensorData(BaseModel):
    signal_type: Literal["cur", "vib"]
    values: List[float]
