import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.config.settings import settings
from app.services.azure_storage import azure_storage
from app.services.spring_client import spring_client
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
            # Azure Storage 연결
            await azure_storage.connect()

            # 헬스 체크 (스프링부트 포함)
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
            print(f"📊 스프링부트 서비스: {settings.spring_boot_service_url}")
            print("-" * 60)

        except Exception as e:
            print(f"❌ 스케줄러 시작 실패: {e}")
            raise

    async def stop(self):
        """스케줄러 중지"""
        if not self.is_running:
            print("⚠️ 스케줄러가 이미 실행 중이 아닙니다.")
            return

        print("🔄 시뮬레이터 중지 중...")

        # 실행 중인 작업들이 완료될 수 있도록 잠시 대기
        try:
            self.scheduler.shutdown(wait=True)
        except Exception as e:
            print(f"⚠️ 스케줄러 종료 중 오류: {e}")

        try:
            await azure_storage.disconnect()
        except Exception as e:
            print(f"⚠️ Azure Storage 연결 해제 중 오류: {e}")

        self.is_running = False
        print("🛑 시뮬레이터 중지됨")

    async def _initial_health_check(self):
        """초기 헬스 체크 (스프링부트 서비스 포함)"""
        print("🔍 서비스 헬스 체크 중...")

        # 스프링부트 서비스 헬스 체크
        spring_health = await spring_client.health_check()
        status = "✅" if spring_health else "❌"
        print(
            f"   {status} Spring Boot Service ({settings.spring_boot_service_url})")

        if not spring_health:
            print("⚠️ 경고: 스프링부트 서비스에 연결할 수 없습니다. 시뮬레이터는 시작되지만 데이터 전송이 실패할 수 있습니다.")

        print("-" * 60)

    async def _simulate_data_collection(self):
        """주기적 Welding Machine 데이터 수집 및 스프링부트 전송 작업"""
        try:
            print(
                f"🔄 Welding Machine 데이터 수집 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Azure Blob에서 전류 + 진동 데이터 시뮬레이션
            simulated_data = await azure_storage.simulate_real_time_data()

            if not simulated_data:
                print("⚠️ 수집할 데이터가 없습니다.")
                return

            current_data = simulated_data["current"]
            vibration_data = simulated_data["vibration"]

            print(f"📊 전류 데이터: {len(current_data['values'])} 포인트")
            print(f"📊 진동 데이터: {len(vibration_data['values'])} 포인트")

            # 각 신호 타입별로 스프링부트 서비스에 전송
            results = await self._send_data_to_spring_boot(current_data, vibration_data)

            if results:
                print("✅ 데이터 전송 완료")
                for signal_type, result in results.items():
                    print(f"   📡 {signal_type}: {result.get('message', 'OK')}")
            else:
                print("❌ 데이터 전송 실패")

            print("-" * 60)

        except asyncio.CancelledError:
            print("⚠️ 데이터 수집 작업이 취소되었습니다 (스케줄러 종료)")
            # CancelledError는 다시 raise하지 않음 (정상적인 종료)
            return
        except Exception as e:
            print(f"❌ 데이터 수집 중 오류 발생: {e}")
            # 에러 로깅은 유지 (스프링부트 전송 실패 등의 상황을 위해)
            anomaly_logger.log_error("welding-machine-scheduler", str(e))

    async def _send_data_to_spring_boot(self, current_data: dict, vibration_data: dict) -> dict:
        """
        수집된 데이터를 스프링부트 서비스로 전송

        Args:
            current_data: 전류 센서 데이터
            vibration_data: 진동 센서 데이터

        Returns:
            전송 결과 딕셔너리
        """
        results = {}
        timestamp = datetime.now().isoformat()
        machine_id = "WELDING_MACHINE_001"

        try:
            # 전류 데이터 전송
            current_sensor_data = {
                "signal_type": current_data["signal_type"],
                "values": current_data["values"],
                "machine_id": machine_id,
                "timestamp": timestamp
            }

            print(
                f"📤 전류 데이터 전송 중... (signal_type: {current_data['signal_type']})")
            current_result = await spring_client.send_sensor_data(current_sensor_data)
            results["current"] = current_result

            # 잠시 대기 (API 호출 간격)
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                print("⚠️ 작업이 취소되었습니다.")
                return {}

            # 진동 데이터 전송
            vibration_sensor_data = {
                "signal_type": vibration_data["signal_type"],
                "values": vibration_data["values"],
                "machine_id": machine_id,
                "timestamp": timestamp
            }

            print(
                f"📤 진동 데이터 전송 중... (signal_type: {vibration_data['signal_type']})")
            vibration_result = await spring_client.send_sensor_data(vibration_sensor_data)
            results["vibration"] = vibration_result

            return results

        except asyncio.CancelledError:
            print("⚠️ 스프링부트 데이터 전송이 취소되었습니다.")
            return {}
        except Exception as e:
            print(f"❌ 스프링부트 데이터 전송 실패: {str(e)}")
            # 전송 실패시 로컬 로그에 기록 (백업용)
            anomaly_logger.log_error("spring-boot-transmission", {
                "error": str(e),
                "current_data_size": len(current_data.get("values", [])),
                "vibration_data_size": len(vibration_data.get("values", [])),
                "timestamp": timestamp
            })
            return {}

    def get_status(self) -> dict:
        """스케줄러 상태 정보"""
        jobs = self.scheduler.get_jobs()
        return {
            "is_running": self.is_running,
            "interval_minutes": settings.scheduler_interval_minutes,
            "next_run": str(jobs[0].next_run_time) if jobs else None,
            "spring_boot_url": settings.spring_boot_service_url,
            "data_transmission_mode": "spring_boot_api"
        }


# 글로벌 스케줄러 인스턴스
simulator_scheduler = SimulatorScheduler()
