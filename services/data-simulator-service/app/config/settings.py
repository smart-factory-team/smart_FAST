import os
from typing import List, Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure Storage 설정 (민감한 정보)
    azure_connection_string: str
    azure_container_name: str

    # 스케줄러 설정
    scheduler_interval_minutes: int = 1
    batch_size: int = 10

    # 모델 서비스 설정 (환경별로 다를 수 있음)
    painting_process_equipment_url: str = "http://localhost:8001"
    painting_surface_url: str = "http://localhost:8002"
    press_defect_url: str = "http://localhost:8003"
    press_fault_url: str = "http://localhost:8004"
    vehicle_assembly_url: str = "http://localhost:8005"
    welding_machine_url: str = "http://localhost:8006"

    # 로그 설정
    log_directory: str = "logs"
    log_filename: str = "anomaly_detections.json"
    error_log_filename: str = "errors.json"

    # HTTP 클라이언트 설정
    http_timeout: int = 30
    max_retries: int = 3

    @property
    def model_services(self) -> Dict[str, str]:
        """모델 서비스 URL 매핑"""
        return {
            "painting-process-equipment": self.painting_process_equipment_url,
            "painting-surface": self.painting_surface_url,
            "press-defect": self.press_defect_url,
            "press-fault": self.press_fault_url,
            "vehicle-assembly": self.vehicle_assembly_url,
            "welding-machine": self.welding_machine_url
        }

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


# 글로벌 설정 인스턴스
settings = Settings()
