import os
from typing import Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure Storage 설정
    azure_connection_string: str
    azure_container_name: str = "simulator-data"

    # Painting Process Equipment 전용 설정
    painting_data_folder: str = "painting-process-equipment"

    # 스케줄러 설정
    scheduler_interval_minutes: int = 1
    batch_size: int = 10

    # Painting Process Equipment 모델 서비스 설정
    painting_service_url: str = os.getenv("PAINTING_SERVICE_URL", "http://localhost:8001")

    # 로그 설정
    log_directory: str = "logs"
    log_filename: str = "painting_issue_logs.json"
    error_log_filename: str = "painting_errors.json"

    # HTTP 클라이언트 설정
    http_timeout: int = 30
    max_retries: int = 3

    @property
    def model_services(self) -> Dict[str, str]:
        """Painting Process Equipment 모델 서비스 URL"""
        return {
            "painting-process-equipment": self.painting_service_url
        }

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


settings = Settings()
