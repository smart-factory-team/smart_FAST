import logging
import sys
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pythonjsonlogger import jsonlogger

from app.config.settings import settings


class LevelBasedFormatter(logging.Formatter):
    """로그 레벨에 따라 다른 포맷을 적용"""

    # 기본 포맷
    simple_format = "✅ [%(levelname)s] %(message)s"
    detailed_format = "🚨 [%(levelname)s]:\n" "   └─ %(message)s"

    def __init__(self):
        super().__init__(fmt="%(levelname)s: %(message)s", datefmt=None, style="%")

    def format(self, record):
        if record.levelno >= logging.WARNING:
            self._style._fmt = self.detailed_format
        else:
            self._style._fmt = self.simple_format

        return super().format(record)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        if not log_record.get("timestamp"):
            log_record["timestamp"] = datetime.fromtimestamp(record.created).isoformat()

        # name을 service_name으로 변경하고 고정값 설정
        log_record["service_name"] = "press fault detection"


def setup_loggers():
    """
    애플리케이션의 로거를 설정하고 구성.
    콘솔 핸들러: 설정된 로그 레벨(INFO) 이상의 모든 로그를 콘솔에 출력.
    파일 핸들러: WARNING 레벨 이상의 로그를 JSON 형식으로 파일에 저장
    """

    # 로그 디렉토리 생성
    if not os.path.exists(settings.LOG_DIR):
        os.makedirs(settings.LOG_DIR, exist_ok=True)

    # 1. 시스템 로거 설정 ("system")
    system_logger = logging.getLogger("system")
    system_logger.setLevel(settings.LOG_LEVEL.upper())
    system_logger.propagate = False  # 다른 로거로 이벤트가 전파되지 않도록 설정

    if not system_logger.hasHandlers():
        system_console_handler = logging.StreamHandler(sys.stdout)
        system_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        system_console_handler.setFormatter(system_formatter)
        system_logger.addHandler(system_console_handler)

    # 2. 예측 결과 로거 설정 ("prediction")
    prediction_logger = logging.getLogger("prediction")
    prediction_logger.setLevel(settings.LOG_LEVEL.upper())
    prediction_logger.propagate = False

    if not prediction_logger.hasHandlers():
        # 콘솔 핸들러 - 정상/고장 모두 출력
        prediction_console_handler = logging.StreamHandler(sys.stdout)
        prediction_console_handler.setFormatter(LevelBasedFormatter())
        prediction_logger.addHandler(prediction_console_handler)

        # 파일 핸들러
        # 이상 감지 시에만 로그 파일에 저장
        json_file_handler = TimedRotatingFileHandler(
            filename=settings.ERROR_LOG_FILE_PATH,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )

        json_formatter = CustomJsonFormatter(
            "%(timestamp)s %(service_name)s %(message)s"
        )
        json_file_handler.setFormatter(json_formatter)
        json_file_handler.setLevel(logging.WARNING)
        prediction_logger.addHandler(json_file_handler)

    # return logger


setup_loggers()

system_log = logging.getLogger("system")
prediction_log = logging.getLogger("prediction")

system_log.info("로거 설정이 완료되었습니다.")
