import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 서비스 기본 정보
    service_name: str = "Painting Process Data Simulator"
    service_version: str = "1.0.0"
    service_description: str = "자동차 부품 프레스 구멍 검사 데이터 시뮬레이터"
    
    # Azure Storage 설정
    azure_connection_string: str = os.getenv("AZURE_CONNECTION_STRING")
    azure_container_name: str = "simulator-data"
    azure_press_defect_path: str = "press-defect"
    
    # 모델 서비스 설정
    model_service_url: str = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:8000")
    model_service_predict_endpoint: str = "/predict/inspection"
    model_service_health_endpoint: str = "/health"
    model_service_timeout: int = 300  # 5분 (21장 이미지 처리)
    
    # 스케줄러 설정
    scheduler_interval_seconds: int = 60  # 1분마다 실행
    max_inspection_count: int = 79  # inspection_001 ~ inspection_079
    start_inspection_id: int = 1
    
    # HTTP 클라이언트 설정
    http_timeout: int = 300  # 5분
    http_retries: int = 3
    http_retry_delay: int = 5  # 5초
    
    # 로그 설정
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_file_name: str = "simulator.log"
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    # 시뮬레이션 설정
    simulation_enabled: bool = False  # 기본적으로 비활성화
    current_inspection_id: int = 1
    total_simulations: int = 0
    successful_simulations: int = 0
    failed_simulations: int = 0
    
    # 개발/운영 모드
    debug_mode: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    class Config:
        case_sensitive = False
        env_file = ".env"

# 전역 설정 인스턴스
settings = Settings()

# 설정 정보 출력용 함수
def get_settings_summary():
    """설정 정보 요약 (민감한 정보 제외)"""
    return {
        "service": {
            "name": settings.service_name,
            "version": settings.service_version,
            "description": settings.service_description
        },
        "azure_storage": {
            "container_name": settings.azure_container_name,
            "press_defect_path": settings.azure_press_defect_path,
            "connection_configured": bool(settings.azure_connection_string)
        },
        "model_service": {
            "url": settings.model_service_url,
            "predict_endpoint": settings.model_service_predict_endpoint,
            "timeout": settings.model_service_timeout
        },
        "scheduler": {
            "interval_seconds": settings.scheduler_interval_seconds,
            "max_inspection_count": settings.max_inspection_count,
            "enabled": settings.simulation_enabled
        },
        "logging": {
            "level": settings.log_level,
            "directory": settings.log_dir,
            "debug_mode": settings.debug_mode
        }
    }

# 환경 변수 검증
def validate_settings():
    """필수 설정 검증"""
    errors = []
    
    if not settings.azure_connection_string:
        errors.append("AZURE_CONNECTION_STRING이 설정되지 않았습니다.")
    
    if not settings.model_service_url:
        errors.append("MODEL_SERVICE_URL이 설정되지 않았습니다.")
    
    if settings.scheduler_interval_seconds < 10:
        errors.append("스케줄러 간격은 최소 10초 이상이어야 합니다.")
    
    if settings.max_inspection_count < 1:
        errors.append("inspection 개수는 최소 1개 이상이어야 합니다.")
    
    return errors

# 설정 업데이트 함수들
def update_simulation_status(enabled: bool):
    """시뮬레이션 상태 업데이트"""
    settings.simulation_enabled = enabled

def increment_simulation_stats(success: bool):
    """시뮬레이션 통계 업데이트"""
    settings.total_simulations += 1
    if success:
        settings.successful_simulations += 1
    else:
        settings.failed_simulations += 1

def update_current_inspection_id(inspection_id: int):
    """현재 처리 중인 inspection ID 업데이트"""
    settings.current_inspection_id = inspection_id

def get_simulation_stats():
    """시뮬레이션 통계 반환"""
    success_rate = 0
    if settings.total_simulations > 0:
        success_rate = (settings.successful_simulations / settings.total_simulations) * 100
    
    return {
        "total_simulations": settings.total_simulations,
        "successful_simulations": settings.successful_simulations,
        "failed_simulations": settings.failed_simulations,
        "success_rate": round(success_rate, 2),
        "current_inspection_id": settings.current_inspection_id,
        "simulation_enabled": settings.simulation_enabled
    }