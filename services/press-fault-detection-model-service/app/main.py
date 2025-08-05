import logging
from fastapi import FastAPI
from app.core.model_cache import model_cache
from contextlib import asynccontextmanager
from app.models.model_loader import load_model, load_scaler, load_threshold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행
    try:
        model_cache["model"] = load_model()
        model_cache["scaler"] = load_scaler()
        model_cache["threshold"] = load_threshold()
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

