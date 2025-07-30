from fastapi import FastAPI
from app.routers import predict_router

app = FastAPI(
    title="Robot Welding Anomaly Detection API",
    version="1.0",
    description="API for detecting anomalies in robotic welding machines",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(predict_router.router, prefix="/api")
