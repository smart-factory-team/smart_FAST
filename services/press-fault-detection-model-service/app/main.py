import datetime
from datetime import timezone
import logging
from fastapi import FastAPI
from app.core.model_cache import model_cache
from contextlib import asynccontextmanager
from app.models.model_loader import load_all
from app.routers import predict_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행
    try:
        artifacts = load_all()
        model_cache.update(artifacts)
        logger.info("모델 로딩 완료")
    except Exception as e:
        logger.error(f"모델 초기화 실패: {str(e)}")
        raise RuntimeError(f"Failed to initialize models: {str(e)}") from e
    yield
    
    logger.info("서버 종료")
    model_cache.clear()
    
app = FastAPI(
    title = "프레스 설비(유압 펌프) 고장 예측 API",
    description="상/하부 진동 및 전류의 시계열 데이터를 입력받아 유압 펌프의 고장 여부를 예측합니다.",
    lifespan=lifespan
)

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Hello World"}

app.include_router(predict_router.router)

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.datetime.now(timezone.utc)}
 
@app.get("/ready")
async def ready():
    models_loaded = all(key in model_cache for key in ["model", "scaler", "threshold"])
    return {"status": "ready" if models_loaded else "not ready", "models_loaded": models_loaded}
 
@app.get("/startup")  
async def startup():
    return {"status": "started", "initialization_complete": True}