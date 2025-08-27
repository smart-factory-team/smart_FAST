from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, field_validator  

class Settings(BaseSettings):
    # Azure Storage 설정
    azure_connection_string: str
    azure_container_name: str = "simulator-data"

    # Painting Process Equipment 전용 설정
    painting_data_folder: str = "painting-process-equipment"

    # 스케줄러 설정
    scheduler_interval_seconds: int = 30
    batch_size: int = 10

    # Backend 서비스 설정
    backend_service_url: AnyHttpUrl = "http://localhost:8088/equipment-data" 

    # HTTP 클라이언트 설정
    http_timeout: int = 30

    # 로그 디렉토리
    log_directory: str = "logs"

    # Validators  
    @field_validator("scheduler_interval_seconds")
    @classmethod
    def _positive_interval(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("scheduler_interval_seconds must be > 0")
        return v

    @field_validator("batch_size", "http_timeout")
    @classmethod
    def _positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size and http_timeout must be > 0")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


settings = Settings()
