from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.routers import simulator_router
from app.utils.logger import system_log
from app.routers.simulator_router import scheduler_service


@asynccontextmanager
async def lifespan(app: FastAPI):

    system_log.info(f"애플리케이션 시작 준비 완료.")
    yield
    system_log.info("서비스 종료 중...")
    if scheduler_service.is_running:
        await scheduler_service.stop_simulation()
    system_log.info(f"애플리케이션이 종료됩니다.")


app = FastAPI(title="유압 펌프 고장 예측 시뮬레이터", description="", lifespan=lifespan)

# 라우터 등록
app.include_router(simulator_router.router)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Press Fault Data Simulator Service",
        "version": "1.0.0",
        "description": "유압 펌프 고장 예측 시뮬레이터 서비스",
    }


@app.get("/health")
async def health():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """준비 상태 체크 엔드포인트"""
    return {"status": "ready"}


@app.get("/startup")
async def startup():
    """시작 상태 체크 엔드포인트"""
    return {"status": "started"}
