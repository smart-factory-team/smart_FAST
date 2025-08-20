import aiohttp
import asyncio
import time
from datetime import datetime
from typing import Optional

from app.config.settings import settings
from app.models.data_models import PredictionRequest
from app.utils.logger import system_log


class SpringBootService:
    """Spring Boot API 호출 서비스 (비동기)"""
    
    def __init__(self):
        self.api_url = f"{settings.SPRING_BOOT_BASE_URL}/pressFaultDetectionLogs/data"
        self.timeout = 30  # 30초 타임아웃
        self.max_retries = 3  # 최대 재시도 횟수
        
        system_log.info(f"Spring Boot Service 초기화 완료 - URL: {self.api_url}")
    
    async def send_sensor_data(self, request: PredictionRequest, data_source: str = None) -> bool:
        """
        Spring Boot로 센서 데이터를 전송
        
        Args:
            request: 센서 데이터 (PredictionRequest 객체)
            data_source: 데이터 소스 (파일명 등)
            
        Returns:
            bool: 전송 성공 여부
        """
        
        # 요청 데이터를 JSON 형태로 변환
        request_data = {
            "AI0_Vibration": request.AI0_Vibration,
            "AI1_Vibration": request.AI1_Vibration,
            "AI2_Current": request.AI2_Current,
            "timestamp": datetime.now().isoformat(),
            "source": data_source or "simulator",
            "data_length": len(request.AI0_Vibration)
        }
        
        system_log.debug(f"Spring Boot 전송 데이터 크기: AI0({len(request_data['AI0_Vibration'])}) "
                        f"AI1({len(request_data['AI1_Vibration'])}) "
                        f"AI2({len(request_data['AI2_Current'])})")
        
        # max_retries 횟수(3)만큼 시도
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                timeout = aiohttp.ClientTimeout(total=self.timeout)

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        url=self.api_url,
                        json=request_data,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status in [200, 201, 202]:
                            system_log.info(f"Spring Boot 전송 성공 ({response_time:.3f}s) - "
                                            f"Status: {response.status}, Source: {data_source}")
                            return True
                        
                        response_text = await response.text()
                        system_log.error(f"Spring Boot 전송 실패 (시도 {attempt}/{self.max_retries}) - "
                                    f"Status: {response.status}, Response: {response_text[:200]}")

            except asyncio.TimeoutError:
                system_log.error(f"Spring Boot 전송 타임아웃 (시도 {attempt}/{self.max_retries}) - URL: {self.api_url}")
            
            except aiohttp.ClientConnectionError:
                system_log.error(f"Spring Boot 서버 연결 실패 (시도 {attempt}/{self.max_retries}) - URL: {self.api_url}")
            
            except Exception as e:
                system_log.error(f"예상치 못한 오류 (시도 {attempt}/{self.max_retries}): {str(e)}")

            # 마지막 시도가 아니면 지수 백오프 후 재시도
            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)
            
        return False
        
        return False
    
    async def health_check(self) -> bool:
        """
        Spring Boot 서버 상태 확인
        
        Returns:
            bool: 서버 상태 (True: 정상, False: 비정상)
        """
        try:
            # Spring Boot 기본 루트 경로로 간단한 연결 테스트
            base_url = f"{settings.SPRING_BOOT_BASE_URL}/pressFaultDetectionLogs"
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(base_url) as response:
                    
                    # 연결만 되면 OK (404여도 서버는 살아있음)
                    if response.status in [200, 404]:
                        system_log.info("Spring Boot 서버 상태: 정상")
                        return True
                    else:
                        system_log.warning(f"Spring Boot 서버 상태: 비정상 (Status: {response.status})")
                        return False
                        
        except Exception as e:
            system_log.error(f"Spring Boot 서버 상태 확인 실패: {str(e)}")
            return False