from fastapi import FastAPI
from app.routers import predict_router
from datetime import datetime, timezone

app = FastAPI(
    title="Robot Welding Anomaly Detection API",
    version="1.0"
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
