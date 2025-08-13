from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # 기본 설정
    debug: bool = False
    log_level: str = "INFO"

    # Azure 설정 (환경변수에서 로드)
    azure_connection_string: Optional[str] = None
    azure_container_name: Optional[str] = None

    # API 설정
    model_api_timeout: int = 30
    max_concurrent_requests: int = 10

    class Config:
        env_file = ".env"

settings = Settings()