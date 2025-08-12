from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.config.settings import settings
from app.services.azure_storage import azure_storage
from app.services.backend_client import backend_client
import logging

logger = logging.getLogger(__name__)

class SimulatorScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False

    async def start(self):
        """스케줄러 시작"""
        if self.is_running:
            logger.warning("⚠️ 스케줄러가 이미 실행 중입니다.")
            return

        try:
            self.scheduler.add_job(
                func=self._simulate_and_send_data,
                trigger=IntervalTrigger(seconds=settings.scheduler_interval_seconds),
                id='data_simulation',
                name='Data Simulation and Sending',
                replace_existing=True
            )

            self.scheduler.start()
            self.is_running = True

            logger.info(f"🚀 백엔드 데이터 전송 시뮬레이터 시작! (간격: {settings.scheduler_interval_seconds}초)")
            logger.info(f"🎯 대상 백엔드 서비스: {settings.backend_service_url}")

        except Exception as e:
            logger.error(f"❌ 스케줄러 시작 실패: {e}")
            raise

    async def stop(self):
        """스케줄러 중지"""
        if not self.is_running:
            logger.warning("⚠️ 스케줄러가 실행 중이 아닙니다.")
            return

        self.scheduler.shutdown()
        self.is_running = False
        logger.info("🛑 시뮬레이터 중지됨")

    async def _simulate_and_send_data(self):
        """주기적 데이터 시뮬레이션 및 백엔드 전송"""
        try:
            logger.info(f"🔄 데이터 시뮬레이션 및 전송 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            simulated_data = await azure_storage.simulate_real_time_data()

            if not simulated_data:
                logger.warning("⚠️ 전송할 데이터가 없습니다.")
                return

            await backend_client.send_to_backend(simulated_data, settings.backend_service_url)

        except Exception as e:
            logger.error(f"❌ 데이터 전송 중 오류 발생: {e}")

    def get_status(self) -> dict:
        """스케줄러 상태 정보"""
        jobs = self.scheduler.get_jobs()
        next_run = None
        if jobs:
            next_run = str(jobs[0].next_run_time)
            
        return {
            "is_running": self.is_running,
            "interval_seconds": settings.scheduler_interval_seconds,
            "next_run": next_run,
            "backend_service_url": settings.backend_service_url
        }

simulator_scheduler = SimulatorScheduler()