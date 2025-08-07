from fastapi import FastAPI
from app.routers import predict as predict_router
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from app.services.inference import PaintingSurfaceDefectDetectionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행
    try:
        global detection_service
        detection_service = PaintingSurfaceDefectDetectionService()
        await detection_service.load_model()
        
        # 전역 변수로 라우터에서 접근할 수 있도록 설정
        predict_router.detection_service = detection_service
        
        print(f"[{datetime.now()}] Painting Surface Defect Detection Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize models: {str(e)}") from e
    yield
    # 앱 종료 시 실행
    if detection_service:
        await detection_service.cleanup()


app = FastAPI(
    title="Painting Surface Defect Detection API",
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
