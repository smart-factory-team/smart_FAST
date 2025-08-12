from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
from datetime import datetime

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.exceptions import setup_exception_handlers
from app.core.middleware import setup_middleware
from app.core.dependencies import set_model_manager

# 라우터 임포트
from app.routers import root, health, predict, model

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# 전역 변수로 모델 저장
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 이벤트 핸들러"""

    # Startup
    logger.info("=== 의장 공정 부품 불량 탐지 API 서버 시작 ===")
    logger.info(f"환경: {settings.ENVIRONMENT}")
    logger.info(f"디버그 모드: {settings.DEBUG}")

    try:
        from app.services.model_manager import DefectDetectionModelManager
        from app.core.dependencies import set_model_manager

        model_manager = DefectDetectionModelManager()
        await model_manager.load_model()
        await set_model_manager(model_manager)

        app.state.model_manager = model_manager

    except Exception as e:
        logger.error(f"서버 시작 중 오류: {e}")

    logger.info("서버 시작 완료")

    yield  # 앱 실행

    # Shutdown
    logger.info("서버 종료 중...")
    # Shutdown
    from app.core.dependencies import cleanup_dependencies
    await cleanup_dependencies()
    logger.info("서버 종료 완료")


# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.DEBUG else None,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
    # OpenAPI 태그 정의
    openapi_tags=[
        {
            "name": "Root",
            "description": "서비스 기본 정보"
        },
        {
            "name": "Health",
            "description": "상태 체크 및 모니터링"
        },
        {
            "name": "Prediction",
            "description": "AI 모델 예측 서비스"
        },
        {
            "name": "Model",
            "description": "모델 정보 및 관리"
        }
    ]
)

# 미들웨어 설정
setup_middleware(app)

# 예외 핸들러 설정
setup_exception_handlers(app)

# === 라우터 등록 ===

# 1. 루트 라우터 (/) - 서비스 정보
app.include_router(root.router)

# 2. 헬스체크 라우터 (/health, /ready, /startup)
app.include_router(health.router)

# 3. 예측 라우터 (/predict, /predict/file)
app.include_router(predict.router)

# 4. 모델 라우터 (/model/info, /model/classes 등)
app.include_router(model.router)


# 개발 환경에서만 직접 실행 가능
if __name__ == "__main__":
    logger.info("개발 서버 직접 실행")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
        access_log=True
    )