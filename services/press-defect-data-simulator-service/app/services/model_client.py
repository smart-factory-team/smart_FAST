import aiohttp
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
import json

from config.settings import settings
from utils.logger import simulator_logger

class ModelServiceClient:
    """FastAPI 모델 서비스 클라이언트"""
    
    def __init__(self):
        self.base_url = settings.model_service_url.rstrip('/')
        self.predict_endpoint = settings.model_service_predict_endpoint
        self.health_endpoint = settings.model_service_health_endpoint
        self.timeout = settings.model_service_timeout
        self.retries = settings.http_retries
        self.retry_delay = settings.http_retry_delay
        
        # 연결 상태
        self.service_available = False
        self.last_health_check = None
        
    async def test_connection(self) -> bool:
        """모델 서비스 연결 테스트"""
        try:
            health_url = f"{self.base_url}{self.health_endpoint}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.service_available = True
                        self.last_health_check = time.time()
                        
                        simulator_logger.log_model_service(
                            True, 
                            f"응답시간: {response.headers.get('x-process-time', 'N/A')}"
                        )
                        return True
                    else:
                        simulator_logger.log_model_service(
                            False, 
                            f"HTTP {response.status}: {await response.text()}"
                        )
                        return False
        
        except aiohttp.ClientError as e:
            simulator_logger.log_model_service(False, f"연결 오류: {str(e)}")
            return False
        except Exception as e:
            simulator_logger.log_model_service(False, f"알 수 없는 오류: {str(e)}")
            return False
    
    async def check_service_health(self) -> Dict[str, Any]:
        """모델 서비스 상세 헬스체크"""
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
    
    async def check_model_ready(self) -> Dict[str, Any]:
        """모델 준비 상태 확인 (/ready 엔드포인트)"""
        try:
            ready_url = f"{self.base_url}/ready"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(ready_url) as response:
                    if response.status == 200:
                        ready_data = await response.json()
                        return {
                            'status': 'ready',
                            'model_loaded': ready_data.get('model_loaded', False),
                            'service_data': ready_data
                        }
                    else:
                        error_data = await response.json() if response.content_type == 'application/json' else await response.text()
                        return {
                            'status': 'not_ready',
                            'error': error_data,
                            'http_status': response.status
                        }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_model_info(self) -> Optional[Dict[str, Any]]:
        """모델 정보 조회"""
        try:
            info_url = f"{self.base_url}/model/info"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(info_url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        simulator_logger.logger.warning(f"모델 정보 조회 실패: HTTP {response.status}")
                        return None
        
        except Exception as e:
            simulator_logger.logger.error(f"모델 정보 조회 오류: {str(e)}")
            return None
    
    async def predict_inspection(self, inspection_id: str, images: List[Dict[str, str]]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """검사 배치 예측 요청"""
        
        start_time = time.time()
        
        # 재시도 로직 포함
        for attempt in range(self.retries + 1):
            try:
                predict_url = f"{self.base_url}{self.predict_endpoint}"
                
                # 요청 데이터 구성
                request_data = {
                    "inspection_id": inspection_id,
                    "images": images
                }
                
                # HTTP 요청
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    
                    simulator_logger.logger.info(
                        f"🤖 모델 서비스 요청: {inspection_id} ({len(images)}장) - 시도 {attempt + 1}/{self.retries + 1}"
                    )
                    
                    async with session.post(
                        predict_url,
                        json=request_data,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        processing_time = time.time() - start_time
                        
                        if response.status == 200:
                            result_data = await response.json()
                            
                            simulator_logger.logger.info(
                                f"✅ 모델 서비스 응답 성공: {inspection_id} ({processing_time:.2f}초)"
                            )
                            
                            return True, result_data, None
                        
                        else:
                            error_text = await response.text()
                            error_msg = f"HTTP {response.status}: {error_text}"
                            
                            simulator_logger.logger.warning(
                                f"⚠️ 모델 서비스 응답 오류: {inspection_id} - {error_msg}"
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
    
    async def predict_single_image(self, image_base64: str, image_name: str = "test") -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """단일 이미지 예측 요청 (테스트용)"""
        try:
            predict_url = f"{self.base_url}/predict/"
            
            request_data = {
                "image_base64": image_base64,
                "image_name": image_name
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    predict_url,
                    json=request_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    
                    if response.status == 200:
                        result_data = await response.json()
                        return True, result_data, None
                    else:
                        error_text = await response.text()
                        return False, None, f"HTTP {response.status}: {error_text}"
        
        except Exception as e:
            return False, None, str(e)
    
    async def check_all_services(self) -> Dict[str, Any]:
        """모든 서비스 상태 종합 확인"""
        try:
            # 1. 기본 헬스체크
            health_result = await self.check_service_health()
            
            # 2. 모델 준비 상태
            ready_result = await self.check_model_ready()
            
            # 3. 모델 정보 (옵션)
            model_info = await self.get_model_info()
            
            # 종합 판정
            overall_status = "healthy"
            if health_result['status'] != 'healthy':
                overall_status = "unhealthy"
            elif ready_result['status'] != 'ready':
                overall_status = "not_ready"
            
            return {
                'overall_status': overall_status,
                'health_check': health_result,
                'ready_check': ready_result,
                'model_info': model_info,
                'service_url': self.base_url,
                'last_check': time.time()
            }
        
        except Exception as e:
            return {
                'overall_status': 'error',
                'error': str(e),
                'service_url': self.base_url,
                'last_check': time.time()
            }
    
    def get_client_status(self) -> Dict[str, Any]:
        """클라이언트 상태 정보"""
        return {
            'service_name': 'Model Service Client',
            'base_url': self.base_url,
            'predict_endpoint': self.predict_endpoint,
            'health_endpoint': self.health_endpoint,
            'timeout': self.timeout,
            'retries': self.retries,
            'service_available': self.service_available,
            'last_health_check': self.last_health_check
        }

# 전역 Model Service 클라이언트 인스턴스
model_service_client = ModelServiceClient()