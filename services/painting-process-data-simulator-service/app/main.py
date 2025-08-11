from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config.settings import settings
from app.services.scheduler_service import simulator_scheduler
from app.routers import simulator_router
from app.routers import connection_test_router
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    print("🚀 Data Simulator Service 시작 중...")

    # 환경 변수 체크
    if not settings.azure_connection_string:
        raise ValueError("AZURE_CONNECTION_STRING 환경 변수가 설정되지 않았습니다. .env 파일을 생성하거나 환경 변수를 설정해주세요.")

    # 로그 디렉토리 생성
    os.makedirs(settings.log_directory, exist_ok=True)

    print(f"📁 로그 디렉토리: {settings.log_directory}")
    print(f"🔧 스케줄러 간격: {settings.scheduler_interval_minutes}분")
    print(f"🎯 대상 서비스 수: {len(settings.model_services)}")

    yield

    # 종료 시
    print("🛑 Data Simulator Service 종료 중...")
    if simulator_scheduler.is_running:
        await simulator_scheduler.stop()

# FastAPI 앱 생성
app = FastAPI(
    title="Painting Process Equipment Data Simulator Service",
    description="도장 공정 설비 결함 탐지 모델을 위한 실시간 데이터 시뮬레이터",
    version="1.0.0",
    lifespan=lifespan
)

# 라우터 설정
# 시뮬레이터 활성화/비활성화/상태확인 API 모음
app.include_router(simulator_router.router, prefix="/simulator")
# azure storage 연결, model serving 서비스 연결 확인 API 모음
app.include_router(connection_test_router.router, prefix="/test")


# 아래는 서비스 기본 정보 확인과 서비스 헬스 체크 api 정의
@app.get("/")
async def root():
    """서비스 정보"""
    return {
        "service": "Painting Process Equipment Data Simulator Service",
        "version": "1.0.0",
        "status": "running",
        "target_model": "painting-process-equipment-defect-detection",
        "scheduler_status": simulator_scheduler.get_status()
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}
