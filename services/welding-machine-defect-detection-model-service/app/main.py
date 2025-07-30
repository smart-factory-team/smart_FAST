from fastapi import FastAPI
from app.routers import predict_router

app = FastAPI(
    title="Robot Welding Anomaly Detection API",
    version="1.0"
)

app.include_router(predict_router.router, prefix="/api")
