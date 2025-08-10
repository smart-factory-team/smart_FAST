from pydantic import BaseModel
from typing import Dict, Optional


class PredictionResponse(BaseModel):
    prediction: str
    reconstruction_error: Optional[float]
    is_fault: bool
    attribute_errors: Optional[Dict[str, Optional[float]]] = None
