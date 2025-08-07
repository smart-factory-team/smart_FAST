"""
미들웨어 모듈

FastAPI 애플리케이션의 미들웨어들을 정의하고 설정합니다.
- CORS 설정
- 요청/응답 로깅
- 처리 시간 측정
- 보안 헤더 추가
- 요청 제한 (Rate Limiting)
"""

import time
import uuid
import logging
from typing import Callable, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import settings


logger = logging.getLogger(__name__)


# === 요청 ID 미들웨어 ===

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    각 요청에 고유한 ID를 할당하는 미들웨어

    요청 추적, 로깅, 디버깅에 활용됩니다.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 요청 ID 생성 (기존 헤더가 있으면 사용)
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # request.state에 저장
        request.state.request_id = request_id

        # 다음 미들웨어/핸들러 호출
        response = await call_next(request)

        # 응답 헤더에 요청 ID 추가
        response.headers["X-Request-ID"] = request_id

        return response


# === 처리 시간 측정 미들웨어 ===

class ProcessingTimeMiddleware(BaseHTTPMiddleware):
    """
    요청 처리 시간을 측정하고 로깅하는 미들웨어

    성능 모니터링과 최적화에 활용됩니다.
    """

    def __init__(self, app, log_slow_requests: bool = True, slow_threshold: float = 1.0):
        super().__init__(app)
        self.log_slow_requests = log_slow_requests
        self.slow_threshold = slow_threshold  # 느린 요청 임계값 (초)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 시작 시간 기록
        start_time = time.time()
        request.state.start_time = start_time

        # 다음 미들웨어/핸들러 호출
        response = await call_next(request)

        # 처리 시간 계산
        processing_time = time.time() - start_time

        # 응답 헤더에 처리 시간 추가
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}"

        # 느린 요청 로깅
        if self.log_slow_requests and processing_time > self.slow_threshold:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.warning(
                f"느린 요청 감지: {processing_time:.3f}초 - "
                f"{request.method} {request.url.path} "
                f"(Request ID: {request_id})"
            )

        return response


# === 요청/응답 로깅 미들웨어 ===

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    요청과 응답을 로깅하는 미들웨어

    API 사용 패턴 분석과 디버깅에 활용됩니다.
    """

    def __init__(self, app, log_body: bool = False, max_body_size: int = 1024):
        super().__init__(app)
        self.log_body = log_body
        self.max_body_size = max_body_size

        # 로깅하지 않을 경로들
        self.skip_paths = {"/health", "/ready", "/startup", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 스킵할 경로 확인
        if request.url.path in self.skip_paths:
            return await call_next(request)

        request_id = getattr(request.state, "request_id", "unknown")

        # 요청 로깅
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params) if request.query_params else None,
            "client_ip": client_ip,
            "user_agent": user_agent[:100],  # User-Agent 길이 제한
            "content_type": request.headers.get("Content-Type"),
            "content_length": request.headers.get("Content-Length"),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"요청 시작: {request.method} {request.url.path}", extra=log_data)

        # 요청 본문 로깅 (선택적)
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body and len(body) <= self.max_body_size:
                    log_data["request_body"] = body.decode("utf-8", errors="ignore")
                elif body:
                    log_data["request_body"] = f"<body too large: {len(body)} bytes>"
            except Exception as e:
                log_data["request_body_error"] = str(e)

        # 다음 미들웨어/핸들러 호출
        try:
            response = await call_next(request)

            # 응답 로깅
            processing_time = getattr(request.state, "start_time", None)
            if processing_time:
                processing_time = time.time() - processing_time

            log_data.update({
                "status_code": response.status_code,
                "response_size": response.headers.get("Content-Length"),
                "processing_time": f"{processing_time:.3f}" if processing_time else None
            })

            log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
            logger.log(
                log_level,
                f"요청 완료: {response.status_code} - {request.method} {request.url.path}",
                extra=log_data
            )

            return response

        except Exception as e:
            # 예외 로깅
            processing_time = getattr(request.state, "start_time", None)
            if processing_time:
                processing_time = time.time() - processing_time

            log_data.update({
                "exception": str(e),
                "exception_type": type(e).__name__,
                "processing_time": f"{processing_time:.3f}" if processing_time else None
            })

            logger.error(
                f"요청 중 예외 발생: {type(e).__name__} - {request.method} {request.url.path}",
                extra=log_data,
                exc_info=True
            )

            raise


# === 보안 헤더 미들웨어 ===

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    보안 관련 HTTP 헤더를 추가하는 미들웨어

    XSS, 클릭재킹 등의 보안 위협을 방지합니다.
    """

    def __init__(self, app):
        super().__init__(app)

        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; img-src 'self' data: https:; script-src 'self' 'unsafe-inline'",
        }

        # HTTPS가 활성화된 경우에만 추가할 헤더들
        if not settings.DEBUG:
            self.security_headers.update({
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            })

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # 보안 헤더 추가
        for header, value in self.security_headers.items():
            response.headers[header] = value

        return response


# === Rate Limiting 미들웨어 ===

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    요청 제한(Rate Limiting)을 구현하는 미들웨어

    API 남용을 방지하고 서버 자원을 보호합니다.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
        enabled: bool = True
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.enabled = enabled

        # 클라이언트별 요청 기록 (실제 환경에서는 Redis 사용 권장)
        self.request_counts = defaultdict(list)
        self.burst_counts = defaultdict(int)

        # 제외할 경로들
        self.excluded_paths = {"/health", "/ready", "/startup"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled or request.url.path in self.excluded_paths:
            return await call_next(request)

        # 클라이언트 식별
        client_id = self._get_client_id(request)
        now = datetime.now()

        # 1분 단위 요청 제한 확인
        minute_ago = now - timedelta(minutes=1)

        # 오래된 요청 기록 정리
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > minute_ago
        ]

        # 현재 분의 요청 수 확인
        current_requests = len(self.request_counts[client_id])

        if current_requests >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": "요청 제한 초과: 분당 최대 요청 수를 초과했습니다",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "details": {
                        "limit": self.requests_per_minute,
                        "window": "1 minute",
                        "retry_after": 60
                    },
                    "timestamp": now.isoformat()
                },
                headers={"Retry-After": "60"}
            )

        # 버스트 제한 확인 (짧은 시간 내 대량 요청)
        recent_requests = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > (now - timedelta(seconds=10))
        ]

        if len(recent_requests) >= self.burst_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": "버스트 제한 초과: 짧은 시간 내 너무 많은 요청",
                    "error_code": "BURST_LIMIT_EXCEEDED",
                    "details": {
                        "burst_limit": self.burst_limit,
                        "window": "10 seconds",
                        "retry_after": 10
                    },
                    "timestamp": now.isoformat()
                },
                headers={"Retry-After": "10"}
            )

        # 요청 기록
        self.request_counts[client_id].append(now)

        # 응답에 제한 정보 헤더 추가
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - current_requests - 1)
        )
        response.headers["X-RateLimit-Reset"] = str(int((now + timedelta(minutes=1)).timestamp()))

        return response

    def _get_client_id(self, request: Request) -> str:
        """클라이언트 식별자 생성"""
        # API 키가 있으면 우선 사용
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key:
            return f"api_key:{api_key[:10]}"  # API 키의 일부만 사용

        # IP 주소 사용
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"ip:{client_ip}"


# === 모델 상태 확인 미들웨어 ===

class ModelHealthMiddleware(BaseHTTPMiddleware):
    """
    모델 상태를 확인하고 필요시 요청을 차단하는 미들웨어

    모델이 로드되지 않은 상태에서의 예측 요청을 방지합니다.
    """

    def __init__(self, app):
        super().__init__(app)

        # 모델 상태 확인이 필요한 경로들
        self.model_required_paths = {"/predict", "/predict/file", "/predict/batch"}

        # 항상 허용할 경로들
        self.always_allowed_paths = {
            "/", "/health", "/ready", "/startup",
            "/docs", "/redoc", "/openapi.json",
            "/model/info", "/model/classes"
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 모델 확인이 필요없는 경로는 통과
        if (request.url.path in self.always_allowed_paths or
            not any(request.url.path.startswith(path) for path in self.model_required_paths)):
            return await call_next(request)

        # 앱 상태에서 모델 매니저 확인

        model_manager = getattr(request.app.state, "model_manager", None)
        logger.info(f"모델 상태 확인중 - model_manager: {model_manager}")

        if not model_manager or not model_manager.is_loaded:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.warning(
                f"모델 미로드 상태에서 예측 요청 차단: {request.url.path} "
                f"(Request ID: {request_id})"
            )

            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "AI 모델이 로드되지 않았습니다. 잠시 후 다시 시도해주세요.",
                    "error_code": "MODEL_NOT_LOADED",
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                }
            )

        return await call_next(request)


# === 미들웨어 설정 함수 ===

def setup_middleware(app: FastAPI) -> None:
    """
    FastAPI 앱에 모든 미들웨어를 설정

    Args:
        app: FastAPI 인스턴스
    """

    # 1. 신뢰할 수 있는 호스트 설정 (프로덕션에서만)
#     if settings.ENVIRONMENT == "production":
#         app.add_middleware(
#             TrustedHostMiddleware,
#             allowed_hosts=["*"]  # 실제 운영에서는 특정 도메인으로 제한
#         )

    # 2. CORS 미들웨어
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=settings.CORS_ORIGINS,
#         allow_credentials=True,
#         allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#         allow_headers=["*"],
#         expose_headers=["X-Request-ID", "X-Processing-Time"]
#     )

    # 3. 보안 헤더 미들웨어
#     app.add_middleware(SecurityHeadersMiddleware)

    # 4. Rate Limiting 미들웨어 (프로덕션에서만)
#     if settings.ENVIRONMENT == "production":
#         app.add_middleware(
#             RateLimitMiddleware,
#             requests_per_minute=100,  # 분당 100회
#             burst_limit=20,  # 10초간 20회
#             enabled=True
#         )

    # 5. 모델 상태 확인 미들웨어
    # app.add_middleware(ModelHealthMiddleware)

    # 6. 로깅 미들웨어
#     app.add_middleware(
#         LoggingMiddleware,
#         log_body=settings.DEBUG,  # 개발환경에서만 본문 로깅
#         max_body_size=1024
#     )

    # 7. 처리 시간 측정 미들웨어
#     app.add_middleware(
#         ProcessingTimeMiddleware,
#         log_slow_requests=True,
#         slow_threshold=2.0  # 2초 이상 소요되는 요청 로깅
#     )

    # 8. 요청 ID 미들웨어 (가장 먼저 실행되도록 마지막에 추가)
#     app.add_middleware(RequestIDMiddleware)
#
#     logger.info("미들웨어 설정 완료")


# === 미들웨어 상태 조회 ===

def get_middleware_status() -> Dict[str, Any]:
    """
    미들웨어 상태 정보 반환

    Returns:
        Dict: 미들웨어 상태 정보
    """
    return {
        "enabled_middleware": [
            "RequestIDMiddleware",
            "ProcessingTimeMiddleware",
            "LoggingMiddleware",
            "SecurityHeadersMiddleware",
            "ModelHealthMiddleware",
            "CORSMiddleware"
        ],
        "conditional_middleware": {
            "TrustedHostMiddleware": settings.ENVIRONMENT == "production",
            "RateLimitMiddleware": settings.ENVIRONMENT == "production"
        },
        "cors_origins": settings.CORS_ORIGINS,
        "security_headers_enabled": True,
        "rate_limiting_enabled": settings.ENVIRONMENT == "production"
    }