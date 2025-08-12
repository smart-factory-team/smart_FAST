from pydantic_settings import BaseSettings


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
    backend_service_url: str = "http://localhost:8087/equipment-data"

    # HTTP 클라이언트 설정
    http_timeout: int = 30

    # 로그 디렉토리
    log_directory: str = "logs"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


settings = Settings()
