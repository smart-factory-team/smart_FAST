from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    reconstruction_error: float
    is_fault: bool