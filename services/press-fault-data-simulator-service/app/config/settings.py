from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field
from urllib.parse import urljoin
import os


class Settings(BaseSettings):
    """
    애플리케이션 설정을 관리하는 클래스.
    .env 파일 또는 환경 변수에서 값을 읽어옵니다.
    """

    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "./logs/press_fault_detections.json"
    ERROR_LOG_FILE_PATH: str = "./logs/press_fault_errors.json"

    # Azure Storage 설정
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_STORAGE_CONTAINER_NAME: str = "simulator-data"
    PRESS_FAULT_FOLDER: str = "press-fault-hydraulic-pump"

    # 모델 API 서버 설정
    PRESS_FAULT_MODEL_BASE_URL: str = "http://127.0.0.1:8004"
    PREDICT_API_ENDPOINT: str = "/predict"

    SIMULATOR_INTERVAL_MINUTES: int = 1

    @computed_field
    @property
    def PREDICTION_API_FULL_URL(self) -> str:
        """
        기본 URL과 endpoint를 조합하여 API URL 생성
        """
        base_url = str(self.PRESS_FAULT_MODEL_BASE_URL)
        return urljoin(base_url, self.PREDICT_API_ENDPOINT)

    @computed_field
    @property
    def LOG_DIR(self) -> str:
        return os.path.dirname(self.ERROR_LOG_FILE_PATH)

    # .env 파일을 읽도록 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env 파일에 정의되지 않은 환경변수는 무시
    )


settings = Settings()
