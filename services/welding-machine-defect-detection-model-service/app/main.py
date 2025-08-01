from fastapi import FastAPI
from app.routers import predict_router
from datetime import datetime, timezone
from app.core.model_cache import model_cache
from contextlib import asynccontextmanager
from app.models.model_loader import load_model_file, load_scaler, load_threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행
    try:
        for signal_type in ["cur", "vib"]:
            model_cache[signal_type]["model"] = load_model_file(signal_type)
            model_cache[signal_type]["scaler"] = load_scaler(signal_type)
            model_cache[signal_type]["threshold"] = load_threshold(signal_type)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize models: {str(e)}") from e
    yield


app = FastAPI(
    title="Robot Welding Anomaly Detection API",
    version="1.0",
    lifespan=lifespan
)

app.include_router(predict_router.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}


@app.get("/ready")
async def ready():
    return {"status": "ready", "models_loaded": True}


@app.get("/startup")
async def startup():
    return {"status": "started", "initialization_complete": True}
