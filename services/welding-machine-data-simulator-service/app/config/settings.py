import os
from typing import List, Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure Storage 설정
    azure_connection_string: str
    azure_container_name: str = "simulator-data"

    # Welding Machine 전용 설정
    welding_data_folder: str = "welding-machine"

    # 스케줄러 설정
    scheduler_interval_minutes: int = 1
    batch_size: int = 10

    # Welding Machine 모델 서비스 설정
    welding_machine_url: str = "http://localhost:8006"

    # 로그 설정
    log_directory: str = "logs"
    log_filename: str = "welding_anomaly_detections.json"
    error_log_filename: str = "welding_errors.json"

    # HTTP 클라이언트 설정
    http_timeout: int = 30
    max_retries: int = 3

    @property
    def model_services(self) -> Dict[str, str]:
        """Welding Machine 모델 서비스 URL"""
        return {
            "welding-machine": self.welding_machine_url
        }

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


settings = Settings()
