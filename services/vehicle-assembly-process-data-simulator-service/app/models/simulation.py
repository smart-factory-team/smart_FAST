from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class SimulationRequest(BaseModel):
    azure_connection_string: str
    container_name: str
    model_api_url: str
    test_mapping_path: str = "test_mapping.json"
    max_concurrent: int = 5
    batch_size: int = 10
    limit: Optional[int] = 10
    image_prefix: str = "assembly-model-simulator/"

class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    message: str

class SimulationStatus(BaseModel):
    simulation_id: str
    status: str
    progress: float
    total_images: int
    processed_images: int
    accuracy: Optional[float] = None
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None

class SimulationResult(BaseModel):
    simulation_id: str
    status: str
    metrics: Dict
    results: List[Dict]
    duration_seconds: float