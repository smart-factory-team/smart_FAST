from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config.settings import settings
from app.services.scheduler_service import simulator_scheduler
from app.routers import simulator_router
from app.routers import test_connection_router
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Data Simulator Service 시작 중...")

    if not settings.azure_connection_string:
        print("⚠️ AZURE_CONNECTION_STRING 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일을 생성하거나 환경 변수를 설정해주세요.")

    os.makedirs(settings.log_directory, exist_ok=True)

    print(f"📁 로그 디렉토리: {settings.log_directory}")
    print(f"🔧 스케줄러 간격: {settings.scheduler_interval_minutes}분")
    print(f"🎯 대상 서비스 수: {len(settings.model_services)}")

    yield

    print("🛑 Data Simulator Service 종료 중...")
    if simulator_scheduler.is_running:
        await simulator_scheduler.stop()


app = FastAPI(
    title="Welding Machine Data Simulator Service",
    description="용접기 결함 탐지 모델을 위한 실시간 데이터 시뮬레이터",
    version="1.0.0",
    lifespan=lifespan
)

# ✅ CORS (개발 환경)
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 프론트 도메인 명시
    allow_credentials=True,     # 쿠키/인증 헤더 사용 시 True
    allow_methods=["*"],        # 또는 ["GET","POST","OPTIONS",...]
    allow_headers=["*"],
)

# ✅ 라우터: 여기에서만 최종 prefix 부여
# simulator_router 내부는 @router.get("/status")처럼 상대 경로만 있어야 함
app.include_router(simulator_router.router, prefix="/simulator")
app.include_router(test_connection_router.router,
                   prefix="/test")


@app.get("/")
async def root():
    return {
        "service": "Welding Machine Data Simulator Service",
        "version": "1.0.0",
        "status": "running",
        "target_model": "welding-machine-defect-detection",
        "scheduler_status": simulator_scheduler.get_status()
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
