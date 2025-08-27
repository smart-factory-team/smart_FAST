from typing import Dict, Any, Optional
from threading import Lock

model_cache: Dict[str, Optional[Any]] = {
    "model": None,
    "scaler": None,
    "threshold": None,
}

cache_lock = Lock()
