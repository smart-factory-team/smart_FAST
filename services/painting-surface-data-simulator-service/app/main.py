from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config.settings import settings
from app.services.scheduler_service import simulator_scheduler
from app.routers import simulator_router
from app.routers import test_connection_router
from app.services.azure_storage import azure_storage
from app.services.model_client import painting_surface_model_client
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    print("🚀 Painting Surface Defect Simulator Service 시작 중...")

    # 환경 변수 체크
    print("🔍 환경 변수 확인 중...")
    print(f"   Azure Connection String: {'✅ 설정됨' if settings.azure_connection_string else '❌ 설정되지 않음'}")
    print(f"   Azure Container: {settings.azure_container_name}")
    print(f"   Painting Data Folder: {settings.painting_data_folder}")
    print(f"   Backend URL: {settings.backend_service_url}")
    
    if not settings.azure_connection_string:
        print("⚠️ AZURE_CONNECTION_STRING 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일을 생성하거나 환경 변수를 설정해주세요.")
        print("   예시: AZURE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...")
    else:
        # Azure Storage 연결 테스트
        try:
            print("🔗 Azure Storage 연결 테스트 중...")
            # 간단한 연결 테스트 - 파일 목록 조회
            test_files = await azure_storage.list_data_files()
            print(f"✅ Azure Storage 연결 성공! ({len(test_files)}개 파일 발견)")
        except Exception as e:
            print(f"❌ Azure Storage 연결 실패: {e}")
            print("   연결 문자열과 계정 키를 확인해주세요.")

    # 로그 디렉토리 생성
    os.makedirs(settings.log_directory, exist_ok=True)

    print(f"📁 로그 디렉토리: {settings.log_directory}")
    print(f"🔧 스케줄러 간격: {settings.scheduler_interval_minutes}분")
    print(f"🎯 대상 서비스: 도장 표면 결함탐지 모델")

    yield

    # 종료 시
    print("🛑 Painting Surface Defect Simulator Service 종료 중...")
    if simulator_scheduler.is_running:
        await simulator_scheduler.stop()


# FastAPI 앱 생성
app = FastAPI(
    title="Painting Surface Defect Simulator Service",
    description="도장 표면 결함 탐지 모델을 위한 실시간 데이터 시뮬레이터",
    version="1.0.0",
    lifespan=lifespan
)

# 라우터 설정
# 시뮬레이터 활성화/비활성화/상태확인 API 모음
app.include_router(simulator_router.router, prefix="/simulator")
# Azure Storage 연결, 모델 서비스 연결 확인 API 모음
app.include_router(test_connection_router.router, prefix="/test")

# 아래는 서비스 기본 정보 확인과 서비스 헬스 체크 api 정의
@app.get("/")
async def root():
    """서비스 정보"""
    return {
        "service": "Painting Surface Data Simulator Service",
        "version": "1.0.0",
        "status": "running",
        "target_model": "painting-surface-defect-detection",
        "scheduler_status": simulator_scheduler.get_status(),
        "azure_storage": {
            "container": settings.azure_container_name,
            "data_folder": settings.painting_data_folder,
            "connection_status": "connected" if hasattr(azure_storage, 'client') and azure_storage.client else "disconnected"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}
