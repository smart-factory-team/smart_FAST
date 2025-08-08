"""
로깅 설정 모듈

애플리케이션의 로깅을 설정하고 관리합니다.
- 파일 및 콘솔 로깅
- 로그 레벨 설정
- 로그 포맷 설정
- 로그 파일 로테이션
- 구조화된 로깅 (JSON)
"""

import os
import sys
import logging
import logging.handlers
from typing import Dict, Any, Optional
from datetime import datetime
import json

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """
    JSON 형태로 로그를 포맷팅하는 커스텀 포맷터

    구조화된 로깅을 통해 로그 분석과 모니터링을 용이하게 합니다.
    """

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON 형태로 포맷팅"""

        # 기본 로그 정보
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if record.exc_info and len(record.exc_info) >= 2:
            try:
                exc_type, exc_value = record.exc_info[0], record.exc_info[1]
                log_data["exception"] = {
                    "type": exc_type.__name__ if exc_type and hasattr(exc_type, '__name__') else str(type(exc_type)),
                    "message": str(exc_value) if exc_value is not None else None,
                    "traceback": self.formatException(record.exc_info)
                }
            except Exception as e:
                log_data["exception"] = {"error": f"Failed to format exception: {str(e)}"}

        # 추가 컨텍스트 정보 (record.extra에서)
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'message'
                }:
                    log_data[key] = value

        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """
    색상이 적용된 콘솔 로그 포맷터

    개발 환경에서 로그 가독성을 향상시킵니다.
    """

    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',      # 청록색
        'INFO': '\033[32m',       # 녹색
        'WARNING': '\033[33m',    # 노란색
        'ERROR': '\033[31m',      # 빨간색
        'CRITICAL': '\033[35m',   # 자주색
        'RESET': '\033[0m'        # 색상 리셋
    }

    def format(self, record: logging.LogRecord) -> str:
        """색상이 적용된 로그 포맷팅"""

        # 색상 적용
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # 로그 레벨에 색상 적용
        colored_level = f"{color}{record.levelname}{reset}"
        record.levelname = colored_level

        # 기본 포맷팅
        formatted = super().format(record)

        return formatted


class RequestContextFilter(logging.Filter):
    """
    요청 컨텍스트 정보를 로그에 추가하는 필터

    요청 ID, 사용자 정보 등을 로그에 자동으로 포함시킵니다.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """로그 레코드에 요청 컨텍스트 정보 추가"""

        # 기본값 설정
        record.request_id = getattr(record, 'request_id', 'N/A')
        record.client_ip = getattr(record, 'client_ip', 'N/A')
        record.user_agent = getattr(record, 'user_agent', 'N/A')

        return True


def setup_logging():
    """
    애플리케이션 로깅 설정

    환경에 따라 적절한 로깅 설정을 구성합니다.
    """

    # 로그 디렉토리 생성
    os.makedirs(settings.LOG_DIR, exist_ok=True)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 1. 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))

    if settings.DEBUG:
        # 개발 환경: 색상 적용된 사람이 읽기 쉬운 포맷
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # 프로덕션 환경: JSON 포맷
        console_formatter = JSONFormatter()

    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(RequestContextFilter())
    root_logger.addHandler(console_handler)

    # 2. 파일 핸들러 설정 (항상 JSON 포맷)

    # 일반 로그 파일 (로테이션)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(settings.LOG_DIR, 'app.log'),
        maxBytes=settings.LOG_FILE_MAX_SIZE,
        backupCount=settings.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())
    file_handler.addFilter(RequestContextFilter())
    root_logger.addHandler(file_handler)

    # 에러 로그 파일 (별도)
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(settings.LOG_DIR, 'error.log'),
        maxBytes=settings.LOG_FILE_MAX_SIZE,
        backupCount=settings.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    error_handler.addFilter(RequestContextFilter())
    root_logger.addHandler(error_handler)

    # 3. 모델 관련 로그 파일 (별도)
    model_logger = logging.getLogger('app.services.model_manager')
    model_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(settings.LOG_DIR, 'model.log'),
        maxBytes=settings.LOG_FILE_MAX_SIZE,
        backupCount=settings.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    model_handler.setLevel(logging.INFO)
    model_handler.setFormatter(JSONFormatter())
    model_handler.addFilter(RequestContextFilter())
    model_logger.addHandler(model_handler)
    model_logger.propagate = False  # 중복 로깅 방지

    # 4. 예측 관련 로그 파일 (별도)
    prediction_logger = logging.getLogger('app.services.prediction_service')
    prediction_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(settings.LOG_DIR, 'prediction.log'),
        maxBytes=settings.LOG_FILE_MAX_SIZE,
        backupCount=settings.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    prediction_handler.setLevel(logging.INFO)
    prediction_handler.setFormatter(JSONFormatter())
    prediction_handler.addFilter(RequestContextFilter())
    prediction_logger.addHandler(prediction_handler)
    prediction_logger.propagate = False

    # 5. 외부 라이브러리 로그 레벨 조정
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    # 로깅 설정 완료 메시지
    logger = logging.getLogger(__name__)
    logger.info(
        f"로깅 설정 완료 - 레벨: {settings.LOG_LEVEL}, "
        f"환경: {settings.ENVIRONMENT}, "
        f"디렉토리: {settings.LOG_DIR}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    지정된 이름의 로거 반환

    Args:
        name: 로거 이름

    Returns:
        logging.Logger: 설정된 로거
    """
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context
):
    """
    컨텍스트 정보와 함께 로그 기록

    Args:
        logger: 로거 인스턴스
        level: 로그 레벨
        message: 로그 메시지
        **context: 추가 컨텍스트 정보
    """
    logger.log(level, message, extra=context)


def log_prediction(
    logger: logging.Logger,
    request_id: str,
    method: str,
    processing_time: float,
    result: Dict[str, Any],
    file_info: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
):
    """
    예측 관련 로그 기록

    Args:
        logger: 로거 인스턴스
        request_id: 요청 ID
        method: 예측 방법 (file, json, batch)
        processing_time: 처리 시간
        result: 예측 결과
        file_info: 파일 정보 (선택적)
        error: 오류 메시지 (선택적)
    """
    log_data = {
        "request_id": request_id,
        "prediction_method": method,
        "processing_time": processing_time,
        "timestamp": datetime.now().isoformat()
    }

    if error:
        log_data["error"] = error
        log_data["success"] = False
        logger.error("예측 실패", extra=log_data)
    else:
        log_data.update({
            "success": True,
            "predicted_category": result.get("predicted_category_name"),
            "confidence": result.get("confidence"),
            "is_defective": result.get("is_defective")
        })

        if file_info:
            log_data["file_info"] = file_info

        logger.info("예측 성공", extra=log_data)


def log_model_operation(
    logger: logging.Logger,
    operation: str,
    duration: float,
    success: bool,
    details: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
):
    """
    모델 관련 작업 로그 기록

    Args:
        logger: 로거 인스턴스
        operation: 작업 유형 (load, reload, predict)
        duration: 작업 소요 시간
        success: 성공 여부
        details: 추가 세부 정보
        error: 오류 메시지 (선택적)
    """
    log_data = {
        "operation": operation,
        "duration": duration,
        "success": success,
        "timestamp": datetime.now().isoformat()
    }

    if details:
        log_data.update(details)

    if error:
        log_data["error"] = error
        logger.error(f"모델 작업 실패: {operation}", extra=log_data)
    else:
        logger.info(f"모델 작업 성공: {operation}", extra=log_data)


def setup_access_logging():
    """
    액세스 로그 설정 (Uvicorn용)

    HTTP 요청/응답 로그를 별도로 관리합니다.
    """

    # 액세스 로그 파일 핸들러
    access_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(settings.LOG_DIR, 'access.log'),
        maxBytes=settings.LOG_FILE_MAX_SIZE,
        backupCount=settings.LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )

    access_formatter = logging.Formatter(
        '%(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s %(content_length)s "%(referer)s" "%(user_agent)s" %(process_time).3f'
    )
    access_handler.setFormatter(access_formatter)

    # Uvicorn 액세스 로거에 추가
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addHandler(access_handler)


class PerformanceLogger:
    """
    성능 메트릭 로깅을 위한 클래스

    응답 시간, 처리량 등의 성능 지표를 추적합니다.
    """

    def __init__(self):
        self.logger = logging.getLogger("performance")
        self.metrics = {
            "request_count": 0,
            "total_processing_time": 0.0,
            "error_count": 0
        }

    def log_request(self, processing_time: float, success: bool, endpoint: str):
        """요청 성능 로그 기록"""
        self.metrics["request_count"] += 1
        self.metrics["total_processing_time"] += processing_time

        if not success:
            self.metrics["error_count"] += 1

        # 성능 로그 기록
        self.logger.info(
            "Request performance",
            extra={
                "processing_time": processing_time,
                "success": success,
                "endpoint": endpoint,
                "average_time": self.metrics["total_processing_time"] / self.metrics["request_count"],
                "error_rate": self.metrics["error_count"] / self.metrics["request_count"],
                "total_requests": self.metrics["request_count"]
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """현재 성능 메트릭 반환"""
        if self.metrics["request_count"] > 0:
            return {
                "total_requests": self.metrics["request_count"],
                "average_processing_time": self.metrics["total_processing_time"] / self.metrics["request_count"],
                "error_rate": self.metrics["error_count"] / self.metrics["request_count"],
                "total_processing_time": self.metrics["total_processing_time"],
                "total_errors": self.metrics["error_count"]
            }
        return {"total_requests": 0}

    def reset_metrics(self):
        """메트릭 초기화"""
        self.metrics = {
            "request_count": 0,
            "total_processing_time": 0.0,
            "error_count": 0
        }


# 전역 성능 로거 인스턴스
performance_logger = PerformanceLogger()


def get_performance_metrics() -> Dict[str, Any]:
    """성능 메트릭 조회"""
    return performance_logger.get_metrics()


def reset_performance_metrics():
    """성능 메트릭 초기화"""
    performance_logger.reset_metrics()


def log_startup_info():
    """서비스 시작 정보 로깅"""
    logger = logging.getLogger(__name__)

    startup_info = {
        "service_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "debug_mode": settings.DEBUG,
        "log_level": settings.LOG_LEVEL,
        "model_name": settings.MODEL_NAME,
        "use_gpu": settings.USE_GPU,
        "startup_time": datetime.now().isoformat()
    }

    logger.info("=== 서비스 시작 ===", extra=startup_info)


def log_shutdown_info():
    """서비스 종료 정보 로깅"""
    logger = logging.getLogger(__name__)

    shutdown_info = {
        "shutdown_time": datetime.now().isoformat(),
        "final_metrics": get_performance_metrics()
    }

    logger.info("=== 서비스 종료 ===", extra=shutdown_info)


# 로그 분석을 위한 유틸리티 함수들

def analyze_log_file(log_file_path: str, hours: int = 24) -> Dict[str, Any]:
    """
    로그 파일 분석 (기본 통계)

    Args:
        log_file_path: 로그 파일 경로
        hours: 분석할 시간 범위 (시간)

    Returns:
        Dict: 로그 분석 결과
    """
    try:
        import json
        from collections import defaultdict, Counter
        from datetime import datetime, timedelta

        now = datetime.now()
        cutoff_time = now - timedelta(hours=hours)

        stats = {
            "total_lines": 0,
            "level_counts": Counter(),
            "error_types": Counter(),
            "request_counts": defaultdict(int),
            "processing_times": [],
            "time_range": {
                "start": cutoff_time.isoformat(),
                "end": now.isoformat()
            }
        }

        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    log_time = datetime.fromisoformat(log_entry.get("timestamp", ""))

                    if log_time < cutoff_time:
                        continue

                    stats["total_lines"] += 1
                    stats["level_counts"][log_entry.get("level", "UNKNOWN")] += 1

                    # 에러 타입 분석
                    if log_entry.get("level") == "ERROR":
                        error_type = log_entry.get("exception", {}).get("type", "Unknown")
                        stats["error_types"][error_type] += 1

                    # 요청 관련 통계
                    if "processing_time" in log_entry:
                        try:
                            proc_time = float(log_entry["processing_time"])
                            stats["processing_times"].append(proc_time)
                        except (ValueError, TypeError):
                            pass

                    # 엔드포인트별 요청 수
                    if "path" in log_entry:
                        stats["request_counts"][log_entry["path"]] += 1

                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

        # 처리 시간 통계 계산
        if stats["processing_times"]:
            import statistics
            proc_times = stats["processing_times"]
            stats["processing_time_stats"] = {
                "count": len(proc_times),
                "avg": statistics.mean(proc_times),
                "median": statistics.median(proc_times),
                "min": min(proc_times),
                "max": max(proc_times),
                "p95": statistics.quantiles(proc_times, n=100)[94] if len(proc_times) > 20 else max(proc_times)
            }

        return stats

    except Exception as e:
        return {"error": f"로그 분석 중 오류 발생: {str(e)}"}


def get_recent_errors(log_file_path: str, count: int = 10) -> list:
    """
    최근 에러 로그 조회

    Args:
        log_file_path: 로그 파일 경로
        count: 조회할 에러 수

    Returns:
        list: 최근 에러 로그들
    """
    errors = []

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 역순으로 검사하여 최근 에러부터 수집
        for line in reversed(lines):
            try:
                log_entry = json.loads(line.strip())
                if log_entry.get("level") in ["ERROR", "CRITICAL"]:
                    errors.append(log_entry)
                    if len(errors) >= count:
                        break
            except json.JSONDecodeError:
                continue

    except Exception as e:
        errors.append({"error": f"에러 로그 조회 실패: {str(e)}"})

    return errors