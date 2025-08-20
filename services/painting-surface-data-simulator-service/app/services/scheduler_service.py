from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.config.settings import settings
from app.services.azure_storage import azure_storage
from app.services.model_client import painting_surface_model_client
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
            print("🔧 스케줄러 시작 준비 중...")
            
            # 헬스 체크
            await self._initial_health_check()

            # 스케줄 작업 등록
            print("📅 스케줄 작업 등록 중...")
            self.scheduler.add_job(
                func=self._simulate_data_collection,
                trigger=IntervalTrigger(
                    minutes=settings.scheduler_interval_minutes),
                id='data_simulation',
                name='Data Collection Simulation',
                replace_existing=True
            )
            print("✅ 스케줄 작업 등록 완료")

            print("🚀 스케줄러 시작 중...")
            self.scheduler.start()
            self.is_running = True

            print(f"🚀 시뮬레이터 시작! (간격: {settings.scheduler_interval_minutes}분)")
            print(f"📊 대상 서비스: 도장 표면 결함탐지 모델")
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
        await azure_storage.disconnect()
        self.is_running = False
        print("🛑 시뮬레이터 중지됨")

    async def _initial_health_check(self):
        """초기 헬스 체크"""
        print("🔍 백엔드 서비스 헬스 체크 중...")
        is_healthy = await painting_surface_model_client.health_check()

        status = "✅" if is_healthy else "❌"
        print(f"   {status} 백엔드 서비스")

        if not is_healthy:
            raise Exception("백엔드 서비스가 비활성 상태입니다.")

        print(f"📈 활성 서비스: 1/1")
        print("-" * 60)

    async def _simulate_data_collection(self):
        """주기적 도장 표면 결함 감지 데이터 수집 및 예측 작업"""
        try:
            print(f"🔄 도장 표면 결함 감지 데이터 수집 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("📊 Azure Storage에서 이미지 데이터 조회 중...")

            # Azure Blob에서 도장 표면 이미지 데이터 시뮬레이션
            simulated_data = await azure_storage.simulate_painting_surface_data()

            if not simulated_data:
                print("⚠️ 수집할 데이터가 없습니다.")
                return

            image_data = simulated_data["images"]
            print(f"📊 이미지 데이터: {len(image_data)} 개")

            # 백엔드 서비스에 결함 감지 요청
            print("🤖 백엔드 서비스에 결함 감지 요청 중...")
            predictions = await painting_surface_model_client.predict_painting_surface_data(image_data)

            if not predictions:
                print("❌ 예측 결과를 받을 수 없습니다.")
                return

            # 결과 처리
            combined_result = predictions.get("combined")
            if not combined_result:
                print("❌ 조합된 예측 결과가 없습니다.")
                return

            # 이상 감지 여부에 따른 로깅
            if combined_result.get("status") == "anomaly":
                # 전체 예측 정보와 원본 데이터 함께 로깅
                anomaly_logger.log_anomaly(
                    "painting-surface",  # 도장 표면 서비스로 수정
                    combined_result,
                    {
                        "image_data": image_data,
                        "detailed_results": predictions,
                        "simulation_data": simulated_data
                    }
                )
                print("🚨 이상 감지!")
            else:
                anomaly_logger.log_normal_processing(
                    "painting-surface", combined_result)  # 도장 표면 서비스로 수정
                print("✅ 정상 상태")

            # 상세 결과 출력
            print(f"📋 {combined_result.get('combined_logic', 'N/A')}")
            print("-" * 60)

        except Exception as e:
            print(f"❌ 데이터 수집 중 오류 발생: {e}")
            anomaly_logger.log_error("painting-surface-scheduler", str(e))  # 도장 표면 서비스로 수정

    def get_status(self) -> dict:
        """스케줄러 상태 정보"""
        jobs = self.scheduler.get_jobs()
        return {
            "is_running": self.is_running,
            "interval_minutes": settings.scheduler_interval_minutes,
            "next_run": str(jobs[0].next_run_time) if jobs else None,
            "target_service": "painting-surface-defect-detection"
        }


# 글로벌 스케줄러 인스턴스
simulator_scheduler = SimulatorScheduler()
