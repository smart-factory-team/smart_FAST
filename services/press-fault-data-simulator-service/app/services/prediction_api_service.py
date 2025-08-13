import requests
import time
from typing import Optional
import aiohttp
import asyncio
import json

from app.config.settings import settings
from app.models.data_models import PredictionRequest, PredictionResult
from app.utils.logger import system_log


class PredictAPIService:
    """Press Fault Detection Model API 호출 서비스"""

    def __init__(self):
        self.api_url = settings.PREDICTION_API_FULL_URL
        self.timeout = 30
        self.max_retries = 3

        system_log.info(f"Predict API Service 초기화 완료 - URL: {self.api_url}")

    async def call_predict_api(
        self, request: PredictionRequest
    ) -> Optional[PredictionResult]:
        """
        /predict API를 호출하여 예측 결과를 받아옴
        Args:
            request 예측 요청 데이터 (PredictionRequest)
        Returns:
            Optional[PredictionResult]: 예측 결과 또는 None (실패 시)
        """

        request_data = request.model_dump()
        system_log.debug(
            f"API 요청 데이터 크기: AI0({len(request_data['AI0_Vibration'])}) "
            f"AI1({len(request_data['AI1_Vibration'])}) "
            f"AI2({len(request_data['AI2_Current'])})"
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()

                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # POST 요청 전송
                    async with session.post(
                        url=self.api_url,
                        json=request_data,
                        headers={"Content-Type": "application/json"},
                    ) as response:

                        response_time = time.time() - start_time

                        # 응답 상태 확인
                        if response.status == 200:
                            response_json = await response.json()
                            try:
                                result = PredictionResult(**response_json)
                                system_log.info(
                                    f"API 호출 성공 ({response_time:.3f}s) - "
                                    f"Prediction: {result.prediction}, "
                                    f"Is_fault: {result.is_fault}"
                                )

                                return result
                            except Exception as e:
                                system_log.error(f"응답 데이터 검증 실패: {str(e)}")
                                return None
                        else:
                            response_text = await response.text()
                            system_log.error(
                                f"API 호출 실패 (시도 {attempt}/{self.max_retries}) - "
                                f"Status: {response.status}, "
                                f"Response: {response_text[:200]}"
                            )
                            # 마지막 시도가 아니면 재시도
                            if attempt < self.max_retries:
                                await asyncio.sleep(2**attempt)
                                continue
                            else:
                                return None

            except asyncio.TimeoutError:
                system_log.error(
                    f"API 호출 타임아웃 (시도 {attempt}/{self.max_retries}) - "
                    f"URL: {self.api_url}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    return None

            except aiohttp.ClientConnectionError:
                system_log.error(
                    f"API 서버 연결 실패 (시도 {attempt}/{self.max_retries}) - "
                    f"URL: {self.api_url}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    return None

            except Exception as e:
                system_log.error(
                    f"예상치 못한 오류 (시도 {attempt}/{self.max_retries}): {str(e)}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    return None
        return None

    async def health_check(self) -> bool:
        """
        API 서버 상태 확인

        Returns:
            bool: 서버 상태 (True: 정상, False: 비정상)
        """
        try:
            base_url = str(settings.PRESS_FAULT_MODEL_BASE_URL)
            health_url = f"{base_url}/health"

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as response:

                    if response.status == 200:
                        system_log.info("API 서버 상태: 정상")
                        return True
                    else:
                        system_log.warning(
                            f"API 서버 상태: 비정상 (Status: {response.status})"
                        )
                        return False

        except requests.exceptions.RequestException as e:
            system_log.error(f"API 서버 상태 확인 실패: {str(e)}")
            return False
