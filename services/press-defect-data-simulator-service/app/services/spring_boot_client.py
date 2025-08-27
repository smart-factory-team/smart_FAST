import aiohttp
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
import json

from app.config.settings import settings
from app.utils.logger import simulator_logger

class SpringBootServiceClient:
    """Spring Boot 서비스 클라이언트 - Event Driven Architecture"""
    
    def __init__(self):
        self.base_url = settings.spring_boot_service_url.rstrip('/')
        self.raw_data_endpoint = settings.spring_boot_raw_data_endpoint
        self.health_endpoint = settings.spring_boot_health_endpoint
        self.timeout = settings.spring_boot_timeout
        self.retries = settings.http_retries
        self.retry_delay = settings.http_retry_delay
        
        # 연결 상태
        self.service_available = False
        self.last_health_check = None
        
    async def test_connection(self) -> bool:
        """Spring Boot 서비스 연결 테스트"""
        try:
            health_url = f"{self.base_url}{self.health_endpoint}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.service_available = True
                        self.last_health_check = time.time()
                        
                        simulator_logger.logger.info(
                            f"✅ Spring Boot 서비스 연결 성공: {self.base_url}"
                        )
                        return True
                    else:
                        simulator_logger.logger.warning(
                            f"⚠️ Spring Boot 서비스 응답 오류: HTTP {response.status}"
                        )
                        return False
        
        except aiohttp.ClientError as e:
            simulator_logger.logger.error(f"❌ Spring Boot 서비스 연결 오류: {str(e)}")
            return False
        except Exception as e:
            simulator_logger.logger.error(f"❌ Spring Boot 서비스 알 수 없는 오류: {str(e)}")
            return False
    
    async def check_service_health(self) -> Dict[str, Any]:
        """Spring Boot 서비스 상세 헬스체크"""
        try:
            health_url = f"{self.base_url}{self.health_endpoint}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                start_time = time.time()
                async with session.get(health_url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        health_data = await response.json()
                        
                        return {
                            'status': 'healthy',
                            'response_time': round(response_time, 3),
                            'service_data': health_data,
                            'timestamp': time.time()
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'response_time': round(response_time, 3),
                            'error': f"HTTP {response.status}",
                            'timestamp': time.time()
                        }
        
        except Exception as e:
            return {
                'status': 'unreachable',
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def send_raw_data(self, inspection_id: str, images: List[Dict[str, str]]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        # 디버깅용 로그 추가
        simulator_logger.logger.info(f"🔍 DEBUG: send_raw_data 시작 - {inspection_id}, 이미지 수: {len(images)}")
        simulator_logger.logger.info(f"🔍 DEBUG: URL: {self.base_url}{self.raw_data_endpoint}")
        simulator_logger.logger.info(f"🔍 DEBUG: 타임아웃: {self.timeout}초")  

        """Spring Boot로 원시 데이터 전송 (Event Driven 방식)"""
        
        start_time = time.time()
        
        # 재시도 로직 포함
        for attempt in range(self.retries + 1):
            try:
                raw_data_url = f"{self.base_url}{self.raw_data_endpoint}"
                
                # 요청 데이터 구성 - Spring Boot DTO 형식에 맞춤
                request_data = {
                    "inspectionId": inspection_id,
                    "images": images,
                    "source": "simulator",
                    "clientInfo": "painting-process-data-simulator-service",
                    "metadata": f"attempt_{attempt + 1}"
                }
                
                # HTTP 요청
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    
                    simulator_logger.logger.info(
                        f"📤 Spring Boot로 원시 데이터 전송: {inspection_id} ({len(images)}장) - 시도 {attempt + 1}/{self.retries + 1}"
                    )
                    
                    async with session.post(
                        raw_data_url,
                        json=request_data,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        processing_time = time.time() - start_time
                        
                        if response.status == 200:
                            result_data = await response.json()
                            
                            simulator_logger.logger.info(
                                f"✅ Spring Boot 원시 데이터 전송 성공: {inspection_id} ({processing_time:.2f}초)"
                            )
                            
                            return True, result_data, None
                        
                        else:
                            error_text = await response.text()
                            error_msg = f"HTTP {response.status}: {error_text}"
                            
                            simulator_logger.logger.warning(
                                f"⚠️ Spring Boot 응답 오류: {inspection_id} - {error_msg}"
                            )
                            
                            # 5xx 에러는 재시도, 4xx 에러는 즉시 실패
                            if response.status >= 500 and attempt < self.retries:
                                simulator_logger.logger.info(f"🔄 재시도 대기 중... ({self.retry_delay}초)")
                                await asyncio.sleep(self.retry_delay)
                                continue
                            else:
                                return False, None, error_msg
            
            except aiohttp.ClientTimeout:
                error_msg = f"요청 시간 초과 ({self.timeout}초)"
                simulator_logger.logger.warning(f"⏰ {inspection_id}: {error_msg}")
                
                if attempt < self.retries:
                    simulator_logger.logger.info(f"🔄 재시도 대기 중... ({self.retry_delay}초)")
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    return False, None, error_msg
            
            except aiohttp.ClientError as e:
                error_msg = f"네트워크 오류: {str(e)}"
                simulator_logger.logger.warning(f"🌐 {inspection_id}: {error_msg}")
                
                if attempt < self.retries:
                    simulator_logger.logger.info(f"🔄 재시도 대기 중... ({self.retry_delay}초)")
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    return False, None, error_msg
            
            except Exception as e:
                error_msg = f"예상치 못한 오류: {str(e)}"
                simulator_logger.logger.error(f"💥 {inspection_id}: {error_msg}")
                
                if attempt < self.retries:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    return False, None, error_msg
        
        # 모든 재시도 실패
        processing_time = time.time() - start_time
        final_error = f"모든 재시도 실패 ({self.retries + 1}회, {processing_time:.2f}초)"
        return False, None, final_error
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Spring Boot 서비스 상태 조회"""
        try:
            status_url = f"{self.base_url}/api/press-defect/status"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(status_url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        simulator_logger.logger.warning(f"서비스 상태 조회 실패: HTTP {response.status}")
                        return None
        
        except Exception as e:
            simulator_logger.logger.error(f"서비스 상태 조회 오류: {str(e)}")
            return None
    
    def get_client_status(self) -> Dict[str, Any]:
        """클라이언트 상태 정보"""
        return {
            'service_name': 'Spring Boot Service Client',
            'base_url': self.base_url,
            'raw_data_endpoint': self.raw_data_endpoint,
            'health_endpoint': self.health_endpoint,
            'timeout': self.timeout,
            'retries': self.retries,
            'service_available': self.service_available,
            'last_health_check': self.last_health_check,
            'architecture_mode': 'event_driven'
        }

# 전역 Spring Boot Service 클라이언트 인스턴스
spring_boot_client = SpringBootServiceClient()