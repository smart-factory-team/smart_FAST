import asyncio
from typing import Optional
from datetime import datetime

from app.services.azure_storage_service import AzureStorageService
from app.services.prediction_api_service import PredictAPIService
from app.models.data_models import PredictionRequest
from app.config.settings import settings
from app.utils.logger import system_log, prediction_log


class SchedulerService:
    """시뮬레이션 스케줄러 서비스 - 백그라운드에서 1분마다 시뮬레이션 실행"""

    def __init__(self):
        self.is_running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # 서비스 인스턴스들
        self.storage_service = AzureStorageService()
        self.api_service = PredictAPIService()

        # 통계
        self.total_predictions = 0
        self.fault_detections = 0
        self.start_time: Optional[datetime] = None

    async def start_simulation(self) -> bool:
        """시뮬레이션 시작!"""

        if self.is_running:
            system_log.warning("시뮬레이션이 이미 실행 중입니다.")
            return False

        system_log.info("🚀 시뮬레이션 시작 중...")

        # API 서버 상태 확인
        if not await self.api_service.health_check():
            system_log.error("API 서버 연결 실패. 시뮬레이션을 시작할 수 없습니다.")
            return False

        # Azure Storage 연결 확인
        if not await self.storage_service.connect():
            system_log.error("Azure Storage 연결 실패.")
            return False

        self.is_running = True
        self.start_time = datetime.now()
        self.total_predictions = 0
        self.fault_detections = 0

        self.loop = asyncio.get_event_loop()
        self.task = self.loop.create_task(self._run_simulation_loop())

        system_log.info(
            f"✅ 시뮬레이션 시작됨 - 간격: {settings.SIMULATOR_INTERVAL_MINUTES}분"
        )
        system_log.info(f"📊 API URL: {settings.PREDICTION_API_FULL_URL}")

        return True

    async def stop_simulation(self) -> bool:
        """시뮬레이션 종료"""
        if not self.is_running:
            system_log.warning("시뮬레이션이 실행 중이지 않습니다.")
            return False

        system_log.info("⏹️ 시뮬레이션 종료 중...")

        self.is_running = False

        # 태스크 취소 및 정리
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await asyncio.wait_for(self.task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Azure Storage 연결 정리
        await self.storage_service.close()
        # 통계 출력
        if self.start_time:
            duration = datetime.now() - self.start_time
            system_log.info(f"실행 시간: {duration}")
            system_log.info(f"총 예측 횟수: {self.total_predictions}")

        system_log.info("✅ 시뮬레이션 종료 완료")
        return True

    async def _run_simulation_loop(self):
        """시뮬레이션 메인 루프 (백그라운드 스레드에서 실행)"""
        system_log.info("시뮬레이션 루프 시작")

        while self.is_running:
            try:
                # 1회 시뮬레이션 실행
                await self._run_single_simulation()
                # 설정된 간격만큼 대기 (분 단위를 초로 변환)
                interval_seconds = settings.SIMULATOR_INTERVAL_MINUTES * 60
                
                if interval_seconds > 0:
                    # 종료 신호를 확인하면서 대기
                    for _ in range(interval_seconds):
                        if not self.is_running:
                            break
                        await asyncio.sleep(1.0)  # 1초마다 체크
                else:
                    # 간격이 0이어도 이벤트 루프에 제어권을 넘겨 다른 작업이 실행되도록 함
                    await asyncio.sleep(0)

            except asyncio.CancelledError:
                system_log.info("시뮬레이션 루프 취소됨")
                break
            except Exception as e:
                system_log.error(f"시뮬레이션 루프 오류: {str(e)}")
                # 오류 발생 시 잠시 대기 후 재시도
                await asyncio.sleep(10.0)

        system_log.info("시뮬레이션 루프 종료")

    async def _run_single_simulation(self) -> bool:
        """시뮬레이션 1회 실행"""
        try:
            # 1. Azure Storage에서 데이터 수집
            data_result = await self.storage_service.get_next_minute_data()

            if data_result is None:
                system_log.warning("데이터를 가져올 수 없습니다.")
                return False

            minute_data, file_name, is_end_of_file = data_result

            # 2. 예측  요청 데이터 생성
            prediction_request = PredictionRequest.from_csv_data(minute_data)
            # 3. API 호출
            prediction_result = await self.api_service.call_predict_api(
                prediction_request
            )

            if prediction_result is None:
                system_log.error("API 호출 실패")
                return False

            # 4. 결과 로그 처리
            self._handle_prediction_result(prediction_result, file_name)

            # 5. 통계 업데이트
            self.total_predictions += 1
            if prediction_result.is_fault:
                self.fault_detections += 1

            if is_end_of_file:
                system_log.info(f"파일 '{file_name}' 처리 완료")

            return True

        except Exception as e:
            system_log.error(f"시뮬레이션 실행 오류: {str(e)}")
            return False

    def _handle_prediction_result(self, result, data_source: str):
        """예측 결과 처리 (로그 기록)"""
        status = "FAULT DETECTED" if result.is_fault else "✅ NORMAL"

        # fault_probability가 None일 수 있으므로 안전하게 처리
        probability_str = (
            f"{result.fault_probability:.4f}"
            if result.fault_probability is not None
            else "N/A"
        )

        message = (
            f"{status} - Prediction: {result.prediction}, "
            f"Probability: {probability_str}, "
        )

        if result.is_fault:
            # 이상 감지 시 WARNING 레벨 (파일 저장)
            prediction_log.warning(message)
        else:
            # 정상 시 INFO 레벨 (콘솔만)
            prediction_log.info(message)

    def get_simulation_status(self) -> dict:
        """현재 시뮬레이션 상태 반환"""
        if not self.is_running:
            return {"status": "stopped", "is_running": False}

        # 실행 시간 계산
        runtime = None
        if self.start_time:
            runtime = str(datetime.now() - self.start_time)

        # Azure Storage 상태
        storage_status = self.storage_service.get_current_status()

        return {
            "status": "running",
            "is_running": True,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "runtime": runtime,
            "total_predictions": self.total_predictions,
            "fault_detections": self.fault_detections,
            "interval_minutes": settings.SIMULATOR_INTERVAL_MINUTES,
            "api_url": settings.PREDICTION_API_FULL_URL,
            "storage_status": storage_status,
        }


# 전역 스케줄러 서비스 인스턴스
scheduler_service = SchedulerService()
