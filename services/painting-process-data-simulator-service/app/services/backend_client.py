import httpx
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class BackendClient:
    async def send_to_backend(self, data: Dict[str, Any], url: str):
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=data)
                response.raise_for_status()
                logger.info(f"✅ 데이터 전송 성공: {response.status_code}")
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ 데이터 전송 실패: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                logger.error(f"❌ 데이터 전송 중 예외 발생: {e}")

backend_client = BackendClient()