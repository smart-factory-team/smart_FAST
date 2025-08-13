from fastapi import FastAPI
from app.routers import predict as predict_router
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from app.services.inference import PaintingSurfaceDefectDetectionService
from typing import Annotated, Optional
from fastapi import Depends
import uvicorn

# Storage for the service instance
_detection_service: Optional[PaintingSurfaceDefectDetectionService] = None

async def get_detection_service() -> PaintingSurfaceDefectDetectionService:
    if _detection_service is None:
        raise RuntimeError("Detection service not initialized")
    return _detection_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행
    try:
        global _detection_service
        _detection_service = PaintingSurfaceDefectDetectionService()
        await _detection_service.load_model()
        
        # predict.py의 전역 변수 설정
        import app.routers.predict as predict_module
        predict_module.detection_service = _detection_service
        
        print(f"[{datetime.now()}] Painting Surface Defect Detection Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize models: {str(e)}") from e
    yield
    # 앱 종료 시 실행
    if _detection_service:
        await _detection_service.cleanup()


app = FastAPI(
    title="Painting Surface Defect Detection API",
    version="1.0",
    lifespan=lifespan
)

app.include_router(predict_router.router)


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}


@app.get("/ready")
async def ready():
    return {"status": "ready", "models_loaded": True}


@app.get("/startup")
async def startup():
    return {"status": "started", "initialization_complete": True}


if __name__ == "__main__":
    # 시뮬레이터에서 접근할 수 있도록 포트 8002로 설정
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # 모든 인터페이스에서 접근 가능
        port=8002,
        reload=True
    )

