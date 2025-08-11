import httpx
import asyncio
from typing import Dict, Any, Optional
from app.config.settings import settings


class ModelClient:
    def __init__(self):
        self.timeout = httpx.Timeout(settings.http_timeout)
        self.max_retries = settings.max_retries

    async def predict(self, service_name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """특정 모델 서비스에 예측 요청"""
        service_url = settings.model_services.get(service_name)

        if not service_url:
            print(f"❌ 알 수 없는 서비스: {service_name}")
            return None

        predict_url = f"{service_url}/api/predict"

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(predict_url, json=data)

                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ {service_name} 예측 성공 (시도 {attempt + 1})")
                        return result
                    else:
                        print(
                            f"⚠️ {service_name} HTTP {response.status_code} (시도 {attempt + 1})")

            except httpx.TimeoutException:
                print(f"⏰ {service_name} 타임아웃 (시도 {attempt + 1})")
            except httpx.ConnectError:
                print(f"🔌 {service_name} 연결 실패 (시도 {attempt + 1})")
            except Exception as e:
                print(f"❌ {service_name} 예측 오류 (시도 {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 지수 백오프

        print(f"❌ {service_name} 최대 재시도 횟수 초과")
        return None

    async def predict_welding_data(self, current_data: Dict[str, Any], vibration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Welding Machine에 전류 + 진동 데이터 예측 요청"""
        service_name = "welding-machine"
        service_url = settings.model_services.get(service_name)

        if not service_url:
            print(f"❌ Welding Machine 서비스 URL을 찾을 수 없습니다.")
            return None

        predict_url = f"{service_url}/api/predict"

        results = {}

        # 전류 데이터 예측
        print(f"🔌 전류 데이터 예측 요청...")
        current_result = await self._single_predict(predict_url, current_data, "전류")
        results["current"] = current_result

        # 진동 데이터 예측
        print(f"📳 진동 데이터 예측 요청...")
        vibration_result = await self._single_predict(predict_url, vibration_data, "진동")
        results["vibration"] = vibration_result

        # 전체 결과 조합
        combined_result = self._combine_results(
            current_result, vibration_result)
        results["combined"] = combined_result

        return results

    async def _single_predict(self, predict_url: str, data: Dict[str, Any], data_type: str) -> Optional[Dict[str, Any]]:
        """단일 예측 요청"""
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(predict_url, json=data)

                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ {data_type} 예측 성공 (시도 {attempt + 1})")
                        return result
                    else:
                        print(
                            f"⚠️ {data_type} HTTP {response.status_code} (시도 {attempt + 1})")

            except httpx.TimeoutException:
                print(f"⏰ {data_type} 타임아웃 (시도 {attempt + 1})")
            except httpx.ConnectError:
                print(f"🔌 {data_type} 연결 실패 (시도 {attempt + 1})")
            except Exception as e:
                print(f"❌ {data_type} 예측 오류 (시도 {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 지수 백오프

        print(f"❌ {data_type} 최대 재시도 횟수 초과")
        return None

    def _combine_results(self, current_result: Dict[str, Any], vibration_result: Dict[str, Any]) -> Dict[str, Any]:
        """전류 + 진동 결과 조합"""
        if not current_result or not vibration_result:
            return {
                "status": "error",
                "message": "일부 예측 결과를 받을 수 없습니다.",
                "current_status": current_result.get("status") if current_result else "error",
                "vibration_status": vibration_result.get("status") if vibration_result else "error"
            }

        # 둘 중 하나라도 anomaly면 전체 anomaly
        current_status = current_result.get("status", "normal")
        vibration_status = vibration_result.get("status", "normal")

        if current_status == "anomaly" or vibration_status == "anomaly":
            final_status = "anomaly"
        else:
            final_status = "normal"

        return {
            "status": final_status,
            "current": {
                "status": current_status,
                "mae": current_result.get("mae"),
                "threshold": current_result.get("threshold")
            },
            "vibration": {
                "status": vibration_status,
                "mae": vibration_result.get("mae"),
                "threshold": vibration_result.get("threshold")
            },
            "combined_logic": f"전류: {current_status}, 진동: {vibration_status} → 최종: {final_status}"
        }

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
