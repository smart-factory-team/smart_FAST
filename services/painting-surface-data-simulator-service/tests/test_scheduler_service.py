import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from app.services.scheduler_service import SimulatorScheduler


class TestSimulatorScheduler:
    """시뮬레이터 스케줄러 테스트"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        self.scheduler = SimulatorScheduler()

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리"""
        if self.scheduler.is_running:
            # 이벤트 루프가 실행 중일 때만 create_task 사용
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self.scheduler.stop())
            except RuntimeError:
                # 이벤트 루프가 실행 중이 아니면 무시
                pass

    def test_scheduler_initialization(self):
        """스케줄러 초기화 테스트"""
        assert self.scheduler.is_running is False
        assert self.scheduler.scheduler is not None

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    @patch('app.services.scheduler_service.azure_storage')
    async def test_start_success(self, mock_azure_storage, mock_model_client, mock_settings):
        """스케줄러 시작 성공 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
                # Mock 헬스 체크 성공
        mock_model_client.health_check = AsyncMock(return_value=True)

        # Mock Azure Storage 연결
        mock_azure_storage.connect = AsyncMock()
        
        # Mock 스케줄러
        mock_scheduler_instance = MagicMock()
        self.scheduler.scheduler = mock_scheduler_instance
        
        # 스케줄러 시작
        await self.scheduler.start()
        
        assert self.scheduler.is_running is True
        mock_scheduler_instance.add_job.assert_called_once()
        mock_scheduler_instance.start.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    async def test_start_already_running(self, mock_model_client, mock_settings):
        """이미 실행 중인 스케줄러 시작 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # 이미 실행 중으로 설정
        self.scheduler.is_running = True
        
                # Mock 헬스 체크 성공
        mock_model_client.health_check = AsyncMock(return_value=True)

        # Mock Azure Storage 연결
        with patch('app.services.scheduler_service.azure_storage') as mock_azure:
            mock_azure.connect = AsyncMock()
            
            # Mock 스케줄러
            mock_scheduler_instance = MagicMock()
            self.scheduler.scheduler = mock_scheduler_instance
            
            # 스케줄러 시작 시도
            await self.scheduler.start()
            
            # 이미 실행 중이므로 작업이 추가되지 않음
            mock_scheduler_instance.add_job.assert_not_called()

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    async def test_start_health_check_failure(self, mock_model_client, mock_settings):
        """헬스 체크 실패로 인한 시작 실패 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock 헬스 체크 실패
        mock_model_client.health_check = AsyncMock(return_value=False)
        
        # 스케줄러 시작 시도 (헬스 체크 실패로 예외 발생)
        with pytest.raises(Exception, match="도장 표면 결함탐지 서비스가 비활성 상태입니다."):
            await self.scheduler.start()

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    async def test_start_exception_handling(self, mock_model_client, mock_settings):
        """시작 중 예외 처리 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock 헬스 체크 성공
        mock_model_client.health_check = AsyncMock(return_value=True)
        
        # Mock 스케줄러에서 예외 발생
        mock_scheduler_instance = MagicMock()
        mock_scheduler_instance.add_job.side_effect = Exception("Scheduler error")
        self.scheduler.scheduler = mock_scheduler_instance
        
        # 스케줄러 시작 시도 (예외 발생)
        with pytest.raises(Exception, match="Scheduler error"):
            await self.scheduler.start()

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """실행 중이 아닌 스케줄러 중지 테스트"""
        # 실행 중이 아닌 상태
        self.scheduler.is_running = False
        
        # Mock 스케줄러
        mock_scheduler_instance = MagicMock()
        self.scheduler.scheduler = mock_scheduler_instance
        
        # 스케줄러 중지
        await self.scheduler.stop()
        
        # 이미 중지된 상태이므로 shutdown이 호출되지 않음
        mock_scheduler_instance.shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_running(self):
        """실행 중인 스케줄러 중지 테스트"""
        # 실행 중인 상태
        self.scheduler.is_running = True
        
        # Mock 스케줄러
        mock_scheduler_instance = MagicMock()
        self.scheduler.scheduler = mock_scheduler_instance
        
        # Mock Azure Storage 연결 종료
        with patch('app.services.scheduler_service.azure_storage') as mock_azure:
            mock_azure.disconnect = AsyncMock()
            
            # 스케줄러 중지
            await self.scheduler.stop()
            
            assert self.scheduler.is_running is False
            mock_scheduler_instance.shutdown.assert_called_once()
            mock_azure.disconnect.assert_called_once()

    def test_get_status(self):
        """상태 조회 테스트"""
        # Mock 스케줄러
        mock_scheduler_instance = MagicMock()
        mock_scheduler_instance.get_jobs.return_value = []
        self.scheduler.scheduler = mock_scheduler_instance
        
        # 상태 조회
        status = self.scheduler.get_status()
        
        assert "is_running" in status
        assert "interval_minutes" in status
        assert "target_service" in status
        assert status["is_running"] is False

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    @patch('app.services.scheduler_service.azure_storage')
    @patch('app.services.scheduler_service.anomaly_logger')
    async def test_simulate_data_collection_success(self, mock_logger, mock_azure_storage, mock_model_client, mock_settings):
        """데이터 수집 시뮬레이션 성공 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage 데이터 시뮬레이션
        mock_azure_storage.simulate_painting_surface_data = AsyncMock(return_value={
            "images": ["image1.jpg", "image2.jpg"],
            "metadata": {"source": "test"}
        })
        
        # Mock 모델 클라이언트 예측 (실제 코드 로직에 맞춤)
        mock_model_client.predict_painting_surface_data = AsyncMock(return_value={
            "combined": {
                "status": "normal",
                "defect_count": 0,
                "total_count": 2,
                "combined_logic": "정상 처리 완료"
            }
        })
        
        # Mock 로거
        mock_logger.log_normal_processing = MagicMock()
        
        # 데이터 수집 시뮬레이션 실행
        await self.scheduler._simulate_data_collection()
        
        # Azure Storage에서 데이터 시뮬레이션 호출 확인
        mock_azure_storage.simulate_painting_surface_data.assert_called_once()
        
        # 모델 클라이언트 예측 호출 확인
        mock_model_client.predict_painting_surface_data.assert_called_once_with(["image1.jpg", "image2.jpg"])
        
        # 로거 호출 확인
        mock_logger.log_normal_processing.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.azure_storage')
    async def test_simulate_data_collection_no_data(self, mock_azure_storage, mock_settings):
        """데이터가 없는 경우 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage에서 데이터 없음
        mock_azure_storage.simulate_painting_surface_data = AsyncMock(return_value=None)
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            # 데이터 수집 시뮬레이션 실행
            await self.scheduler._simulate_data_collection()
            
            # 경고 메시지 출력 확인
            mock_print.assert_called_with("⚠️ 수집할 데이터가 없습니다.")

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    @patch('app.services.scheduler_service.azure_storage')
    async def test_simulate_data_collection_prediction_failure(self, mock_azure_storage, mock_model_client, mock_settings):
        """예측 실패 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage 데이터 시뮬레이션
        mock_azure_storage.simulate_painting_surface_data = AsyncMock(return_value={
            "images": ["image1.jpg", "image2.jpg"],
            "metadata": {"source": "test"}
        })
        
        # Mock 모델 클라이언트 예측 실패 (실제 코드 로직에 맞춤)
        mock_model_client.predict_painting_surface_data = AsyncMock(return_value=None)
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            # 데이터 수집 시뮬레이션 실행
            await self.scheduler._simulate_data_collection()
            
            # 예측 실패 메시지 출력 확인
            mock_print.assert_called_with("❌ 예측 결과를 받을 수 없습니다.")

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    @patch('app.services.scheduler_service.azure_storage')
    @patch('app.services.scheduler_service.anomaly_logger')
    async def test_simulate_data_collection_anomaly_detected(self, mock_logger, mock_azure_storage, mock_model_client, mock_settings):
        """결함 탐지 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage 데이터 시뮬레이션
        mock_azure_storage.simulate_painting_surface_data = AsyncMock(return_value={
            "images": ["image1.jpg", "image2.jpg"],
            "metadata": {"source": "test"}
        })
        
        # Mock 모델 클라이언트 예측 (결함 탐지 - 실제 코드 로직에 맞춤)
        mock_model_client.predict_painting_surface_data = AsyncMock(return_value={
            "combined": {
                "status": "anomaly",
                "defect_count": 1,
                "total_count": 2,
                "combined_logic": "결함 탐지됨"
            }
        })
        
        # Mock 로거
        mock_logger.log_anomaly = MagicMock()
        
        # 데이터 수집 시뮬레이션 실행
        await self.scheduler._simulate_data_collection()
        
        # 결함 탐지 로그 호출 확인
        mock_logger.log_anomaly.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    @patch('app.services.scheduler_service.azure_storage')
    async def test_simulate_data_collection_exception_handling(self, mock_azure_storage, mock_model_client, mock_settings):
        """예외 처리 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage에서 예외 발생
        mock_azure_storage.simulate_painting_surface_data = AsyncMock(side_effect=Exception("Storage error"))
        
        # Mock 로거
        with patch('app.services.scheduler_service.anomaly_logger') as mock_logger:
            mock_logger.log_error = MagicMock()
            
            # 데이터 수집 시뮬레이션 실행
            await self.scheduler._simulate_data_collection()
            
            # 로거 에러 호출 확인
            mock_logger.log_error.assert_called_once_with("painting-surface-scheduler", "Storage error")

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    async def test_initial_health_check_success(self, mock_model_client, mock_settings):
        """초기 헬스 체크 성공 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock 헬스 체크 성공
        mock_model_client.health_check = AsyncMock(return_value=True)
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            # 초기 헬스 체크 실행
            await self.scheduler._initial_health_check()
            
            # 성공 메시지 출력 확인 (실제 출력 메시지에 맞춤)
            mock_print.assert_any_call("   ✅ 도장 표면 결함탐지 서비스")

    @pytest.mark.asyncio
    @patch('app.services.scheduler_service.settings')
    @patch('app.services.scheduler_service.painting_surface_model_client')
    async def test_initial_health_check_failure(self, mock_model_client, mock_settings):
        """초기 헬스 체크 실패 테스트"""
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock 헬스 체크 실패
        mock_model_client.health_check = AsyncMock(return_value=False)
        
        # 초기 헬스 체크 실행 (예외 발생)
        with pytest.raises(Exception, match="도장 표면 결함탐지 서비스가 비활성 상태입니다."):
            await self.scheduler._initial_health_check()
