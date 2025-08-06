import httpx
import asyncio
from typing import Dict, Any, Optional
from app.config.settings import settings


class ModelClient:
    def __init__(self):
        self.timeout = httpx.Timeout(settings.http_timeout)
        self.max_retries = settings.max_retries

    async def predict_painting_issue(self, data: Dict[str, Any], client: httpx.AsyncClient = None) -> Optional[Dict[str, Any]]:
        """Painting Process Equipment 모델 서비스에 예측 요청"""
        service_name = "painting-process-equipment"
        service_url = settings.model_services.get(service_name)

        if not service_url:
            print(f"❌ 알 수 없는 서비스: {service_name}")
            return None

        predict_url = f"{service_url}/predict/"

        async def _request(client: httpx.AsyncClient):
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(predict_url, json=data)

                    if response.status_code == 200:
                        response_json = response.json()
                        if "predictions" in response_json and response_json["predictions"]:
                            result = response_json["predictions"][0]
                            result['machineId'] = data.get('machineId')
                            result['timeStamp'] = data.get('timeStamp')
                            print(f"✅ {service_name} 예측 성공 (이슈 감지) (시도 {attempt + 1})")
                            return result
                        else:
                            print(f"⚠️ {service_name} 응답 형식이 올바르지 않음 (시도 {attempt + 1})")
                            return None
                    elif response.status_code == 204:
                        print(f"✅ {service_name} 예측 성공 (정상) (시도 {attempt + 1})")
                        return None
                    else:
                        print(f"⚠️ {service_name} HTTP {response.status_code} (시도 {attempt + 1})")

                except httpx.TimeoutException:
                    print(f"⏰ {service_name} 타임아웃 (시도 {attempt + 1})")
                except httpx.ConnectError:
                    print(f"🔌 {service_name} 연결 실패 (시도 {attempt + 1})")
                except Exception as e:
                    print(f"❌ {service_name} 예측 오류 (시도 {attempt + 1}): {e}")

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 30))

            print(f"❌ {service_name} 최대 재시도 횟수 초과")
            return None

        if client:
            return await _request(client)
        else:
            async with httpx.AsyncClient(timeout=self.timeout) as new_client:
                return await _request(new_client)

    async def health_check(self, service_name: str) -> bool:
        """서비스 헬스 체크"""
        service_url = settings.model_services.get(service_name)

        if not service_url:
            return False

        health_url = f"{service_url}/health"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(health_url)
                return response.status_code == 200
        except:
            return False

    async def health_check_all(self) -> Dict[str, bool]:
        """모든 서비스 헬스 체크"""
        tasks = []
        service_names = list(settings.model_services.keys())

        for service_name in service_names:
            task = self.health_check(service_name)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_status = {}
        for service_name, result in zip(service_names, results):
            health_status[service_name] = result if isinstance(
                result, bool) else False

        return health_status


# 글로벌 모델 클라이언트 인스턴스
model_client = ModelClient()
