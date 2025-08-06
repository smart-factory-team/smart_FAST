from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.config.settings import settings
from app.services.azure_storage import azure_storage
from app.services.model_client import model_client
from app.utils.logger import anomaly_logger


class SimulatorScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False

    async def start(self):
        """스케줄러 시작"""
        if self.is_running:
            print("⚠️ 스케줄러가 이미 실행 중입니다.")
            return

        try:
            

            # 헬스 체크
            await self._initial_health_check()

            # 스케줄 작업 등록
            self.scheduler.add_job(
                func=self._simulate_data_collection,
                trigger=IntervalTrigger(
                    minutes=settings.scheduler_interval_minutes),
                id='data_simulation',
                name='Data Collection Simulation',
                replace_existing=True
            )

            self.scheduler.start()
            self.is_running = True

            print(f"🚀 시뮬레이터 시작! (간격: {settings.scheduler_interval_minutes}분)")
            print(f"📊 대상 서비스: {list(settings.model_services.keys())}")
            print("-" * 60)

        except Exception as e:
            print(f"❌ 스케줄러 시작 실패: {e}")
            raise

    async def stop(self):
        """스케줄러 중지"""
        if not self.is_running:
            print("⚠️ 스케줄러가 실행 중이 아닙니다.")
            return

        self.scheduler.shutdown()
        
        self.is_running = False
        print("🛑 시뮬레이터 중지됨")

    async def _initial_health_check(self):
        """초기 헬스 체크"""
        print("🔍 모델 서비스 헬스 체크 중...")
        health_status = await model_client.health_check_all()

        for service_name, is_healthy in health_status.items():
            status = "✅" if is_healthy else "❌"
            print(f"   {status} {service_name}")

        healthy_count = sum(health_status.values())
        total_count = len(health_status)

        if healthy_count == 0:
            raise Exception("모든 모델 서비스가 비활성 상태입니다.")

        print(f"📈 활성 서비스: {healthy_count}/{total_count}")
        print("-" * 60)

    async def _simulate_data_collection(self):
        """주기적 Painting Process Equipment 데이터 수집 및 예측 작업"""
        try:
            print(
                f"🔄 Painting 데이터 수집 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Azure Blob에서 데이터 시뮬레이션
            simulated_data = await azure_storage.simulate_real_time_data()

            if not simulated_data:
                print("⚠️ 수집할 데이터가 없습니다.")
                return

            # 모델 서비스에 예측 요청
            prediction_result = await model_client.predict_painting_issue(simulated_data)

            # 결과 처리
            if prediction_result:
                # 이상 감지 시 로그 기록
                anomaly_logger.log_anomaly(
                    "painting-process-equipment",
                    prediction_result, # 모델이 반환한 로그 데이터
                    simulated_data # 시뮬레이터가 생성한 원본 데이터
                )
                print(f"🚨 이상 감지! - 이슈: {prediction_result.get('issue')}")
            else:
                # 정상 상태
                anomaly_logger.log_normal_processing(
                    "painting-process-equipment",
                    simulated_data
                )
                print("✅ 정상 상태")

            print("-" * 60)

        except Exception as e:
            print(f"❌ 데이터 수집 중 오류 발생: {e}")
            anomaly_logger.log_error("painting-simulator-scheduler", str(e))

    def get_status(self) -> dict:
        """스케줄러 상태 정보"""
        return {
            "is_running": self.is_running,
            "interval_minutes": settings.scheduler_interval_minutes,
            "next_run": str(self.scheduler.get_jobs()[0].next_run_time) if self.scheduler.get_jobs() else None,
            "total_services": len(settings.model_services)
        }


# 글로벌 스케줄러 인스턴스
simulator_scheduler = SimulatorScheduler()
