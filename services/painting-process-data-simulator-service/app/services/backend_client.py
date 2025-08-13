import httpx
import logging
from typing import Any, Dict
from app.config.settings import settings

logger = logging.getLogger(__name__)

class BackendClient:
    async def send_to_backend(self, data: Dict[str, Any], url: str) -> bool:
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.http_timeout)) as client:
            try:
                response = await client.post(url, json=data)
                response.raise_for_status()
                logger.info("✅ 데이터 전송 성공: %s", response.status_code)
                return True
            except httpx.HTTPStatusError as e:
                # Avoid logging full response body to prevent PII leakage
                logger.error("❌ 데이터 전송 실패(HTTP %s)", getattr(e.response, "status_code", "unknown"))
                return False
            except httpx.RequestError as e:
                logger.error("❌ 네트워크 오류로 데이터 전송 실패: %r", e)
                return False
            except Exception:
                logger.exception("❌ 데이터 전송 중 예외 발생")
                return False

backend_client = BackendClient()