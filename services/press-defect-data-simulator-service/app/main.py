import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any
import signal
import sys

from config.settings import settings, validate_settings, get_settings_summary
from utils.logger import simulator_logger
from services.azure_storage import azure_storage_service
from services.model_client import model_service_client
from services.scheduler_service import scheduler_service
from routers import connection_test_router, simulator_router

# 애플리케이션 생명주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행될 코드"""
    
    # 애플리케이션 시작
    simulator_logger.logger.info("=" * 60)
    simulator_logger.logger.info(f"🚀 {settings.service_name} v{settings.service_version} 시작")
    simulator_logger.logger.info("=" * 60)
    
    try:
        # 1. 설정 검증
        validation_errors = validate_settings()
        if validation_errors:
            simulator_logger.logger.error("❌ 설정 검증 실패:")
            for error in validation_errors:
                simulator_logger.logger.error(f"  - {error}")
            raise Exception("필수 설정이 누락되었습니다.")
        
        simulator_logger.logger.info("✅ 설정 검증 완료")
        
        # 2. 서비스 정보 출력
        settings_summary = get_settings_summary()
        simulator_logger.logger.info("📋 서비스 정보:")
        simulator_logger.logger.info(f"  - 서비스명: {settings_summary['service']['name']}")
        simulator_logger.logger.info(f"  - 버전: {settings_summary['service']['version']}")
        simulator_logger.logger.info(f"  - 설명: {settings_summary['service']['description']}")
        simulator_logger.logger.info(f"  - 스케줄 간격: {settings_summary['scheduler']['interval_seconds']}초")
        simulator_logger.logger.info(f"  - 최대 검사 수: {settings_summary['scheduler']['max_inspection_count']}개")
        
        # 3. 외부 서비스 연결 테스트 (초기화는 하지 않음)
        simulator_logger.logger.info("🔍 외부 서비스 연결 확인 중...")
        
        # Azure Storage 연결 테스트
        azure_available = await azure_storage_service.test_connection()
        if azure_available:
            simulator_logger.logger.info("✅ Azure Storage 연결 확인 완료")
        else:
            simulator_logger.logger.warning("⚠️ Azure Storage 연결 실패 - 시뮬레이션 시작 시 재시도됩니다")
        
        # Model Service 연결 테스트
        model_available = await model_service_client.test_connection()
        if model_available:
            simulator_logger.logger.info("✅ Model Service 연결 확인 완료")
        else:
            simulator_logger.logger.warning("⚠️ Model Service 연결 실패 - 시뮬레이션 시작 시 재시도됩니다")
        
        # 4. 시작 완료 로그
        simulator_logger.logger.info("🎉 애플리케이션 시작 완료!")
        simulator_logger.logger.info("📡 API 서버가 준비되었습니다.")
        simulator_logger.logger.info("=" * 60)
        
        yield  # 애플리케이션 실행
        
    except Exception as e:
        simulator_logger.logger.error(f"💥 애플리케이션 시작 실패: {str(e)}")
        raise
    
    finally:
        # 애플리케이션 종료
        simulator_logger.logger.info("=" * 60)
        simulator_logger.logger.info("🛑 애플리케이션 종료 중...")
        
        try:
            # 스케줄러 중지
            if scheduler_service.running:
                simulator_logger.logger.info("⏹️ 스케줄러 중지 중...")
                await scheduler_service.stop_scheduler()
                simulator_logger.logger.info("✅ 스케줄러 중지 완료")
            
            simulator_logger.logger.info("👋 애플리케이션 종료 완료")
            
        except Exception as e:
            simulator_logger.logger.error(f"❌ 종료 중 오류 발생: {str(e)}")
        
        finally:
            simulator_logger.logger.info("=" * 60)

# FastAPI 애플리케이션 생성
app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    description=settings.service_description,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug_mode else None,
    redoc_url="/redoc" if settings.debug_mode else None
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(connection_test_router.router)
app.include_router(simulator_router.router)

# 기본 엔드포인트들
@app.get("/", response_model=Dict[str, Any])
async def root():
    """서비스 기본 정보"""
    settings_summary = get_settings_summary()
    
    return {
        "service": settings_summary['service'],
        "status": "running",
        "endpoints": {
            "docs": "/docs" if settings.debug_mode else "disabled",
            "health": "/health",
            "connection_test": "/connection/test/all",
            "simulator_status": "/simulator/status",
            "simulator_control": "/simulator/start"
        },
        "configuration": {
            "debug_mode": settings.debug_mode,
            "scheduler_enabled": settings_summary['scheduler']['enabled']
        }
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """애플리케이션 헬스체크"""
    try:
        # 기본 애플리케이션 상태
        app_status = {
            "application": "healthy",
            "timestamp": simulator_logger.logger.handlers[0].formatter.formatTime(
                simulator_logger.logger.makeRecord(
                    "health", 20, __file__, 0, "", (), None
                )
            )
        }
        
        # 스케줄러 헬스체크
        scheduler_health = await scheduler_service.health_check()
        
        # 전체 상태 판정
        overall_healthy = scheduler_health.get('healthy', False)
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "application": app_status,
            "services": scheduler_health,
            "components": {
                "scheduler": scheduler_health.get('scheduler_running', False),
                "azure_storage": scheduler_health.get('azure_storage', False),
                "model_service": scheduler_health.get('model_service', False)
            }
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "application": "error",
            "timestamp": "error"
        }

@app.get("/info", response_model=Dict[str, Any])
async def service_info():
    """서비스 상세 정보"""
    try:
        settings_summary = get_settings_summary()
        
        # 런타임 정보
        scheduler_status = scheduler_service.get_scheduler_status()
        
        return {
            "service_info": settings_summary,
            "runtime_status": {
                "scheduler_running": scheduler_status['scheduler_info']['running'],
                "initialization_completed": scheduler_service.initialization_completed,
                "start_time": scheduler_status['scheduler_info']['start_time'],
                "execution_count": scheduler_status['scheduler_info']['execution_count']
            },
            "external_services": {
                "azure_storage": azure_storage_service.get_service_status(),
                "model_service": model_service_client.get_client_status()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서비스 정보 조회 실패: {str(e)}")

@app.get("/version", response_model=Dict[str, str])
async def version_info():
    """버전 정보"""
    return {
        "service_name": settings.service_name,
        "version": settings.service_version,
        "description": settings.service_description
    }

# 전역 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리"""
    simulator_logger.logger.error(f"💥 처리되지 않은 예외: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "서버에서 예상치 못한 오류가 발생했습니다.",
            "detail": str(exc) if settings.debug_mode else "자세한 내용은 로그를 확인하세요."
        }
    )

# 우아한 종료를 위한 시그널 핸들러
def signal_handler(signum, frame):
    """시그널 핸들러 (Ctrl+C 등)"""
    simulator_logger.logger.info(f"🛑 종료 신호 수신: {signum}")
    sys.exit(0)

# 시그널 등록
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 애플리케이션 실행
if __name__ == "__main__":
    try:
        # 개발 모드 설정
        if settings.debug_mode:
            simulator_logger.logger.info("🔧 개발 모드로 실행 중...")
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="info"
            )
        else:
            simulator_logger.logger.info("🚀 운영 모드로 실행 중...")
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=8000,
                log_level="info"
            )
    
    except KeyboardInterrupt:
        simulator_logger.logger.info("👋 사용자에 의해 종료됨")
    except Exception as e:
        simulator_logger.logger.error(f"💥 실행 중 오류: {str(e)}")
    finally:
        simulator_logger.logger.info("🏁 프로그램 종료")