import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import schedule

from config.settings import settings, update_simulation_status, increment_simulation_stats, update_current_inspection_id, get_simulation_stats
from utils.logger import simulator_logger
from services.azure_storage import azure_storage_service
from services.model_client import model_service_client

class SchedulerService:
    """스케줄러 서비스 - 1분마다 데이터 수집 및 예측"""
    
    def __init__(self):
        self.running = False
        self.scheduler_thread = None
        self.current_inspection_id = settings.start_inspection_id
        self.last_execution_time = None
        self.next_execution_time = None
        self.execution_count = 0
        
        # 서비스 상태
        self.azure_ready = False
        self.model_ready = False
        self.initialization_completed = False
        
        # 통계
        self.start_time = None
        self.total_processing_time = 0
        self.average_processing_time = 0
        
    async def initialize_services(self) -> bool:
        """서비스 초기화 및 헬스체크"""
        try:
            simulator_logger.log_scheduler_event("서비스 초기화 시작")
            
            # 1. Azure Storage 초기화
            simulator_logger.logger.info("🔄 Azure Storage 초기화 중...")
            self.azure_ready = await azure_storage_service.initialize()
            
            if not self.azure_ready:
                simulator_logger.log_scheduler_event("Azure Storage 초기화 실패")
                return False
            
            # 2. Model Service 연결 확인
            simulator_logger.logger.info("🔄 Model Service 연결 확인 중...")
            self.model_ready = await model_service_client.test_connection()
            
            if not self.model_ready:
                simulator_logger.log_scheduler_event("Model Service 연결 실패")
                return False
            
            # 3. 모델 준비 상태 확인
            ready_result = await model_service_client.check_model_ready()
            if ready_result['status'] != 'ready':
                simulator_logger.log_scheduler_event(
                    "Model Service 준비 안됨", 
                    {"status": ready_result['status'], "error": ready_result.get('error')}
                )
                self.model_ready = False
                return False
            
            self.initialization_completed = True
            simulator_logger.log_scheduler_event("모든 서비스 초기화 완료")
            
            return True
            
        except Exception as e:
            simulator_logger.log_scheduler_event("초기화 중 오류 발생", {"error": str(e)})
            return False
    
    async def execute_single_simulation(self) -> bool:
        """단일 시뮬레이션 실행"""
        start_time = time.time()
        inspection_id = self.current_inspection_id
        
        try:
            # 현재 inspection ID 업데이트
            update_current_inspection_id(inspection_id)
            
            simulator_logger.log_simulation_start(inspection_id)
            
            # 1. Azure Storage에서 이미지 다운로드
            download_success, images_data = await azure_storage_service.download_inspection_images(inspection_id)
            
            if not download_success or not images_data:
                error_msg = f"이미지 다운로드 실패: inspection_{inspection_id:03d}"
                processing_time = time.time() - start_time
                simulator_logger.log_simulation_failure(inspection_id, error_msg, processing_time)
                increment_simulation_stats(success=False)
                return False
            
            # 2. 모델 서비스에 예측 요청
            inspection_id_str = f"inspection_{inspection_id:03d}"
            
            prediction_success, result_data, error_msg = await model_service_client.predict_inspection(
                inspection_id=inspection_id_str,
                images=images_data
            )
            
            processing_time = time.time() - start_time
            
            if prediction_success and result_data:
                # 성공 로그
                simulator_logger.log_simulation_success(inspection_id, result_data, processing_time)
                
                # 이상 감지 확인
                simulator_logger.log_anomaly_detection(inspection_id, result_data)
                
                # 통계 업데이트
                increment_simulation_stats(success=True)
                self.total_processing_time += processing_time
                self.execution_count += 1
                self.average_processing_time = self.total_processing_time / self.execution_count
                
                return True
            else:
                # 실패 로그
                simulator_logger.log_simulation_failure(inspection_id, error_msg, processing_time)
                increment_simulation_stats(success=False)
                return False
        
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"시뮬레이션 실행 중 예외 발생: {str(e)}"
            simulator_logger.log_simulation_failure(inspection_id, error_msg, processing_time)
            increment_simulation_stats(success=False)
            return False
        
        finally:
            # 다음 inspection ID로 이동 (순환)
            self.current_inspection_id += 1
            if self.current_inspection_id > settings.max_inspection_count:
                self.current_inspection_id = settings.start_inspection_id
                simulator_logger.log_scheduler_event(
                    "Inspection ID 순환 완료", 
                    {"next_id": self.current_inspection_id}
                )
    
    def _schedule_job(self):
        """스케줄 작업 실행 (동기 함수)"""
        if not settings.simulation_enabled:
            return
        
        try:
            # 비동기 함수를 새 이벤트 루프에서 실행
            asyncio.run(self.execute_single_simulation())
            
            # 실행 시간 업데이트
            self.last_execution_time = datetime.now()
            self.next_execution_time = self.last_execution_time + timedelta(seconds=settings.scheduler_interval_seconds)
            
        except Exception as e:
            simulator_logger.logger.error(f"스케줄 작업 실행 중 오류: {str(e)}")
    
    def _run_scheduler(self):
        """스케줄러 실행 (별도 스레드)"""
        try:
            simulator_logger.log_scheduler_event("스케줄러 스레드 시작")
            
            # 스케줄 등록
            schedule.every(settings.scheduler_interval_seconds).seconds.do(self._schedule_job)
            
            while self.running:
                schedule.run_pending()
                time.sleep(1)  # 1초마다 스케줄 확인
            
            simulator_logger.log_scheduler_event("스케줄러 스레드 종료")
            
        except Exception as e:
            simulator_logger.log_scheduler_event("스케줄러 스레드 오류", {"error": str(e)})
    
    async def start_scheduler(self) -> bool:
        """스케줄러 시작"""
        try:
            if self.running:
                simulator_logger.log_scheduler_event("스케줄러가 이미 실행 중")
                return True
            
            # 서비스 초기화
            if not self.initialization_completed:
                init_success = await self.initialize_services()
                if not init_success:
                    return False
            
            # 시뮬레이션 활성화
            update_simulation_status(True)
            
            # 스케줄러 시작
            self.running = True
            self.start_time = datetime.now()
            
            # 스케줄러 스레드 시작
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
            # 첫 실행 시간 설정
            self.next_execution_time = datetime.now() + timedelta(seconds=settings.scheduler_interval_seconds)
            
            simulator_logger.log_scheduler_event(
                "스케줄러 시작 완료", 
                {
                    "interval": settings.scheduler_interval_seconds,
                    "max_inspection": settings.max_inspection_count,
                    "start_inspection": self.current_inspection_id
                }
            )
            
            return True
            
        except Exception as e:
            simulator_logger.log_scheduler_event("스케줄러 시작 실패", {"error": str(e)})
            return False
    
    async def stop_scheduler(self) -> bool:
        """스케줄러 중지"""
        try:
            if not self.running:
                simulator_logger.log_scheduler_event("스케줄러가 이미 중지됨")
                return True
            
            # 시뮬레이션 비활성화
            update_simulation_status(False)
            
            # 스케줄러 중지
            self.running = False
            
            # 스케줄 클리어
            schedule.clear()
            
            # 스레드 종료 대기 (최대 5초)
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            simulator_logger.log_scheduler_event("스케줄러 중지 완료")
            
            return True
            
        except Exception as e:
            simulator_logger.log_scheduler_event("스케줄러 중지 실패", f'{{"error": "{str(e)}"}}')
            return False
    
    async def restart_scheduler(self) -> bool:
        """스케줄러 재시작"""
        simulator_logger.log_scheduler_event("스케줄러 재시작 요청")
        
        # 중지
        stop_success = await self.stop_scheduler()
        if not stop_success:
            return False
        
        # 잠시 대기
        await asyncio.sleep(2)
        
        # 시작
        return await self.start_scheduler()
    
    async def manual_execution(self) -> Dict[str, Any]:
        """수동 실행 (테스트용)"""
        try:
            simulator_logger.log_scheduler_event("수동 실행 시작")
            
            if not self.initialization_completed:
                init_success = await self.initialize_services()
                if not init_success:
                    return {
                        "success": False,
                        "error": "서비스 초기화 실패"
                    }
            
            # 임시로 시뮬레이션 활성화
            original_status = settings.simulation_enabled
            update_simulation_status(True)
            
            try:
                success = await self.execute_single_simulation()
                
                return {
                    "success": success,
                    "inspection_id": self.current_inspection_id - 1,  # 실행 후 증가했으므로 -1
                    "timestamp": datetime.now().isoformat()
                }
            
            finally:
                # 원래 상태로 복원
                update_simulation_status(original_status)
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """스케줄러 상태 정보"""
        simulation_stats = get_simulation_stats()
        
        status = {
            "scheduler_info": {
                "running": self.running,
                "initialization_completed": self.initialization_completed,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
                "next_execution": self.next_execution_time.isoformat() if self.next_execution_time else None,
                "execution_count": self.execution_count,
                "average_processing_time": round(self.average_processing_time, 2)
            },
            "service_status": {
                "azure_ready": self.azure_ready,
                "model_ready": self.model_ready
            },
            "simulation_stats": simulation_stats,
            "settings": {
                "interval_seconds": settings.scheduler_interval_seconds,
                "max_inspection_count": settings.max_inspection_count,
                "current_inspection_id": self.current_inspection_id
            }
        }
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """스케줄러 헬스체크"""
        try:
            # 서비스 상태 재확인
            if self.azure_ready:
                azure_test = await azure_storage_service.test_connection()
                self.azure_ready = azure_test
            
            if self.model_ready:
                model_test = await model_service_client.test_connection()
                self.model_ready = model_test
            
            # 전반적인 건강 상태
            overall_healthy = self.azure_ready and self.model_ready
            
            if self.running and not overall_healthy:
                simulator_logger.log_scheduler_event(
                    "헬스체크 경고", 
                    {"azure_ready": self.azure_ready, "model_ready": self.model_ready}
                )
            
            return {
                "healthy": overall_healthy,
                "azure_storage": self.azure_ready,
                "model_service": self.model_ready,
                "scheduler_running": self.running,
                "initialization_completed": self.initialization_completed,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

# 전역 스케줄러 서비스 인스턴스
scheduler_service = SchedulerService()