import os
from typing import Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure Storage 설정
    azure_connection_string: str
    azure_container_name: str = "simulator-data"
    
    # 도장 표면 결함 감지 전용 설정
    painting_data_folder: str = "painting-surface"
    
    # 스케줄러 설정
    scheduler_interval_minutes: int = 1
    batch_size: int = 10
    
    # 도장 표면 결함 감지 모델 서비스 설정
    # painting_model_url: str = "http://localhost:8002"
    
    # 백엔드 서비스 설정 (새로 추가)
    backend_url: str = "http://smart-factory-painting-backend:8080"
    
    # 로그 설정
    log_directory: str = "logs"
    log_filename: str = "painting_defect_detections.json"
    error_log_filename: str = "painting_defect_detections.json"
    
    # HTTP 클라이언트 설정
    http_timeout: int = 30
    max_retries: int = 3
    
    @property
    def model_service_url(self) -> str:
        """도장 표면 결함 감지 모델 서비스 URL"""
        return self.painting_model_url
    
    @property
    def backend_service_url(self) -> str:
        """백엔드 서비스 URL"""
        return self.backend_url
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


settings = Settings()
