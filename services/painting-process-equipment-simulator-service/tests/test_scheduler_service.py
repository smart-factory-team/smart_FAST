import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.job import Job

# Import the service under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.scheduler_service import SimulatorScheduler, simulator_scheduler


class TestSimulatorScheduler:
    """Test suite for SimulatorScheduler class"""
    
    @pytest.fixture
    def scheduler_instance(self):
        """Create a fresh scheduler instance for each test"""
        return SimulatorScheduler()
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings configuration"""
        with patch('app.services.scheduler_service.settings') as mock_settings:
            mock_settings.scheduler_interval_minutes = 5
            mock_settings.model_services = {
                'service1': {'url': 'http://service1'},
                'service2': {'url': 'http://service2'}
            }
            yield mock_settings
    
    @pytest.fixture
    def mock_azure_storage(self):
        """Mock Azure storage service"""
        with patch('app.services.scheduler_service.azure_storage') as mock_storage:
            yield mock_storage
    
    @pytest.fixture
    def mock_model_client(self):
        """Mock model client service"""
        with patch('app.services.scheduler_service.model_client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def mock_anomaly_logger(self):
        """Mock anomaly logger"""
        with patch('app.services.scheduler_service.anomaly_logger') as mock_logger:
            yield mock_logger

    def test_init_creates_scheduler_instance(self):
        """Test scheduler initialization creates AsyncIOScheduler instance"""
        scheduler = SimulatorScheduler()
        assert isinstance(scheduler.scheduler, AsyncIOScheduler)
        assert scheduler.is_running is False

    def test_init_default_state(self):
        """Test scheduler starts in correct initial state"""
        scheduler = SimulatorScheduler()
        assert scheduler.is_running is False
        assert hasattr(scheduler, 'scheduler')

    @pytest.mark.asyncio
    async def test_start_successful(self, scheduler_instance, mock_settings, mock_model_client, capsys):
        """Test successful scheduler start"""
        mock_model_client.health_check_all = AsyncMock(return_value={
            'service1': True,
            'service2': True
        })
        
        with patch.object(scheduler_instance.scheduler, 'add_job') as mock_add_job, \
             patch.object(scheduler_instance.scheduler, 'start') as mock_start:
            
            await scheduler_instance.start()
            
            assert scheduler_instance.is_running is True
            mock_add_job.assert_called_once()
            mock_start.assert_called_once()
            
            # Verify job configuration
            call_args = mock_add_job.call_args
            assert call_args[1]['id'] == 'data_simulation'
            assert call_args[1]['name'] == 'Data Collection Simulation'
            assert call_args[1]['replace_existing'] is True
            assert isinstance(call_args[1]['trigger'], IntervalTrigger)

    @pytest.mark.asyncio
    async def test_start_already_running(self, scheduler_instance, capsys):
        """Test start when scheduler is already running"""
        scheduler_instance.is_running = True
        
        await scheduler_instance.start()
        
        captured = capsys.readouterr()
        assert "⚠️ 스케줄러가 이미 실행 중입니다." in captured.out

    @pytest.mark.asyncio
    async def test_start_health_check_failure_all_unhealthy(self, scheduler_instance, mock_model_client):
        """Test start fails when all services are unhealthy"""
        mock_model_client.health_check_all = AsyncMock(return_value={
            'service1': False,
            'service2': False
        })
        
        with pytest.raises(Exception, match="모든 모델 서비스가 비활성 상태입니다."):
            await scheduler_instance.start()
        
        assert scheduler_instance.is_running is False

    @pytest.mark.asyncio
    async def test_start_health_check_partial_healthy(self, scheduler_instance, mock_settings, mock_model_client, capsys):
        """Test start succeeds when some services are healthy"""
        mock_model_client.health_check_all = AsyncMock(return_value={
            'service1': True,
            'service2': False,
            'service3': True
        })
        
        with patch.object(scheduler_instance.scheduler, 'add_job'), \
             patch.object(scheduler_instance.scheduler, 'start'):
            
            await scheduler_instance.start()
            
            assert scheduler_instance.is_running is True
            captured = capsys.readouterr()
            assert "📈 활성 서비스: 2/3" in captured.out

    @pytest.mark.asyncio
    async def test_start_exception_handling(self, scheduler_instance, mock_model_client):
        """Test start handles exceptions properly"""
        mock_model_client.health_check_all = AsyncMock(side_effect=Exception("Network error"))
        
        with pytest.raises(Exception, match="Network error"):
            await scheduler_instance.start()
        
        assert scheduler_instance.is_running is False

    @pytest.mark.asyncio
    async def test_stop_successful(self, scheduler_instance, capsys):
        """Test successful scheduler stop"""
        scheduler_instance.is_running = True
        scheduler_instance.scheduler.shutdown = Mock()
        
        await scheduler_instance.stop()
        
        assert scheduler_instance.is_running is False
        scheduler_instance.scheduler.shutdown.assert_called_once()
        
        captured = capsys.readouterr()
        assert "🛑 시뮬레이터 중지됨" in captured.out

    @pytest.mark.asyncio
    async def test_stop_not_running(self, scheduler_instance, capsys):
        """Test stop when scheduler is not running"""
        scheduler_instance.is_running = False
        
        await scheduler_instance.stop()
        
        captured = capsys.readouterr()
        assert "⚠️ 스케줄러가 실행 중이 아닙니다." in captured.out

    @pytest.mark.asyncio
    async def test_initial_health_check_all_healthy(self, scheduler_instance, mock_model_client, capsys):
        """Test initial health check with all services healthy"""
        mock_model_client.health_check_all = AsyncMock(return_value={
            'service1': True,
            'service2': True
        })
        
        await scheduler_instance._initial_health_check()
        
        captured = capsys.readouterr()
        assert "🔍 모델 서비스 헬스 체크 중..." in captured.out
        assert "✅ service1" in captured.out
        assert "✅ service2" in captured.out
        assert "📈 활성 서비스: 2/2" in captured.out

    @pytest.mark.asyncio
    async def test_initial_health_check_mixed_status(self, scheduler_instance, mock_model_client, capsys):
        """Test initial health check with mixed service status"""
        mock_model_client.health_check_all = AsyncMock(return_value={
            'healthy_service': True,
            'unhealthy_service': False
        })
        
        await scheduler_instance._initial_health_check()
        
        captured = capsys.readouterr()
        assert "✅ healthy_service" in captured.out
        assert "❌ unhealthy_service" in captured.out
        assert "📈 활성 서비스: 1/2" in captured.out

    @pytest.mark.asyncio
    async def test_simulate_data_collection_successful_with_anomaly(self, scheduler_instance, mock_azure_storage, mock_model_client, mock_anomaly_logger, capsys):
        """Test data collection simulation with anomaly detection"""
        mock_data = {"temperature": 85, "pressure": 120}
        mock_prediction = {"issue": "high_temperature", "confidence": 0.95}
        
        mock_azure_storage.simulate_real_time_data = AsyncMock(return_value=mock_data)
        mock_model_client.predict_painting_issue = AsyncMock(return_value=mock_prediction)
        
        await scheduler_instance._simulate_data_collection()
        
        mock_azure_storage.simulate_real_time_data.assert_called_once()
        mock_model_client.predict_painting_issue.assert_called_once_with(mock_data)
        mock_anomaly_logger.log_anomaly.assert_called_once_with(
            "painting-process-equipment",
            mock_prediction,
            mock_data
        )
        
        captured = capsys.readouterr()
        assert "🔄 Painting 데이터 수집 시작" in captured.out
        assert "🚨 이상 감지! - 이슈: high_temperature" in captured.out

    @pytest.mark.asyncio
    async def test_simulate_data_collection_successful_normal(self, scheduler_instance, mock_azure_storage, mock_model_client, mock_anomaly_logger, capsys):
        """Test data collection simulation with normal status"""
        mock_data = {"temperature": 65, "pressure": 100}
        
        mock_azure_storage.simulate_real_time_data = AsyncMock(return_value=mock_data)
        mock_model_client.predict_painting_issue = AsyncMock(return_value=None)
        
        await scheduler_instance._simulate_data_collection()
        
        mock_anomaly_logger.log_normal_processing.assert_called_once_with(
            "painting-process-equipment",
            mock_data
        )
        
        captured = capsys.readouterr()
        assert "✅ 정상 상태" in captured.out

    @pytest.mark.asyncio
    async def test_simulate_data_collection_no_data(self, scheduler_instance, mock_azure_storage, mock_model_client, capsys):
        """Test data collection when no data is available"""
        mock_azure_storage.simulate_real_time_data = AsyncMock(return_value=None)
        
        await scheduler_instance._simulate_data_collection()
        
        mock_model_client.predict_painting_issue.assert_not_called()
        
        captured = capsys.readouterr()
        assert "⚠️ 수집할 데이터가 없습니다." in captured.out

    @pytest.mark.asyncio
    async def test_simulate_data_collection_empty_data(self, scheduler_instance, mock_azure_storage, mock_model_client, capsys):
        """Test data collection with empty data"""
        mock_azure_storage.simulate_real_time_data = AsyncMock(return_value={})
        
        await scheduler_instance._simulate_data_collection()
        
        captured = capsys.readouterr()
        assert "⚠️ 수집할 데이터가 없습니다." in captured.out

    @pytest.mark.asyncio
    async def test_simulate_data_collection_azure_storage_exception(self, scheduler_instance, mock_azure_storage, mock_anomaly_logger, capsys):
        """Test data collection handles Azure storage exceptions"""
        mock_azure_storage.simulate_real_time_data = AsyncMock(side_effect=Exception("Storage error"))
        
        await scheduler_instance._simulate_data_collection()
        
        mock_anomaly_logger.log_error.assert_called_once_with(
            "painting-simulator-scheduler", 
            "Storage error"
        )
        
        captured = capsys.readouterr()
        assert "❌ 데이터 수집 중 오류 발생: Storage error" in captured.out

    @pytest.mark.asyncio
    async def test_simulate_data_collection_model_client_exception(self, scheduler_instance, mock_azure_storage, mock_model_client, mock_anomaly_logger, capsys):
        """Test data collection handles model client exceptions"""
        mock_data = {"temperature": 75}
        mock_azure_storage.simulate_real_time_data = AsyncMock(return_value=mock_data)
        mock_model_client.predict_painting_issue = AsyncMock(side_effect=Exception("Model error"))
        
        await scheduler_instance._simulate_data_collection()
        
        mock_anomaly_logger.log_error.assert_called_once_with(
            "painting-simulator-scheduler", 
            "Model error"
        )

    def test_get_status_with_jobs(self, scheduler_instance, mock_settings):
        """Test get_status returns correct status with active jobs"""
        scheduler_instance.is_running = True
        
        mock_job = Mock(spec=Job)
        mock_job.next_run_time = datetime(2024, 1, 1, 12, 0, 0)
        scheduler_instance.scheduler.get_jobs = Mock(return_value=[mock_job])
        
        status = scheduler_instance.get_status()
        
        assert status["is_running"] is True
        assert status["interval_minutes"] == 5
        assert status["next_run"] == "2024-01-01 12:00:00"
        assert status["total_services"] == 2

    def test_get_status_no_jobs(self, scheduler_instance, mock_settings):
        """Test get_status returns correct status with no jobs"""
        scheduler_instance.is_running = False
        scheduler_instance.scheduler.get_jobs = Mock(return_value=[])
        
        status = scheduler_instance.get_status()
        
        assert status["is_running"] is False
        assert status["interval_minutes"] == 5
        assert status["next_run"] is None
        assert status["total_services"] == 2

    def test_get_status_multiple_jobs(self, scheduler_instance, mock_settings):
        """Test get_status with multiple jobs returns first job's next run time"""
        scheduler_instance.is_running = True
        
        mock_job1 = Mock(spec=Job)
        mock_job1.next_run_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_job2 = Mock(spec=Job)
        mock_job2.next_run_time = datetime(2024, 1, 1, 13, 0, 0)
        
        scheduler_instance.scheduler.get_jobs = Mock(return_value=[mock_job1, mock_job2])
        
        status = scheduler_instance.get_status()
        
        assert status["next_run"] == "2024-01-01 12:00:00"


class TestGlobalSchedulerInstance:
    """Test suite for global scheduler instance"""
    
    def test_global_instance_exists(self):
        """Test that global scheduler instance exists"""
        assert simulator_scheduler is not None
        assert isinstance(simulator_scheduler, SimulatorScheduler)

    def test_global_instance_initial_state(self):
        """Test global scheduler instance initial state"""
        assert simulator_scheduler.is_running is False
        assert hasattr(simulator_scheduler, 'scheduler')


class TestSchedulerIntegration:
    """Integration tests for scheduler service"""
    
    @pytest.mark.asyncio
    async def test_start_stop_cycle(self, mock_settings, mock_model_client):
        """Test complete start-stop cycle"""
        scheduler = SimulatorScheduler()
        mock_model_client.health_check_all = AsyncMock(return_value={'service1': True})
        
        with patch.object(scheduler.scheduler, 'add_job'), \
             patch.object(scheduler.scheduler, 'start'), \
             patch.object(scheduler.scheduler, 'shutdown'):
            
            # Start scheduler
            await scheduler.start()
            assert scheduler.is_running is True
            
            # Stop scheduler
            await scheduler.stop()
            assert scheduler.is_running is False

    @pytest.mark.asyncio
    async def test_multiple_start_calls_idempotent(self, mock_settings, mock_model_client):
        """Test multiple start calls are handled gracefully"""
        scheduler = SimulatorScheduler()
        mock_model_client.health_check_all = AsyncMock(return_value={'service1': True})
        
        with patch.object(scheduler.scheduler, 'add_job') as mock_add_job, \
             patch.object(scheduler.scheduler, 'start') as mock_start:
            
            await scheduler.start()
            await scheduler.start()  # Second call should be ignored
            
            # Should only be called once
            assert mock_add_job.call_count == 1
            assert mock_start.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_stop_calls_idempotent(self):
        """Test multiple stop calls are handled gracefully"""
        scheduler = SimulatorScheduler()
        scheduler.scheduler.shutdown = Mock()
        
        await scheduler.stop()  # First call when not running
        await scheduler.stop()  # Second call when not running
        
        # Shutdown should not be called when not running
        scheduler.scheduler.shutdown.assert_not_called()


class TestSchedulerEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_health_check_empty_services(self, scheduler_instance, mock_model_client):
        """Test health check with no services configured"""
        mock_model_client.health_check_all = AsyncMock(return_value={})
        
        with pytest.raises(Exception, match="모든 모델 서비스가 비활성 상태입니다."):
            await scheduler_instance._initial_health_check()

    @pytest.mark.asyncio
    async def test_simulate_data_collection_prediction_result_falsy_values(self, scheduler_instance, mock_azure_storage, mock_model_client, mock_anomaly_logger):
        """Test data collection with various falsy prediction results"""
        mock_data = {"temperature": 70}
        mock_azure_storage.simulate_real_time_data = AsyncMock(return_value=mock_data)
        
        # Test with False
        mock_model_client.predict_painting_issue = AsyncMock(return_value=False)
        await scheduler_instance._simulate_data_collection()
        mock_anomaly_logger.log_normal_processing.assert_called()
        
        # Test with empty dict
        mock_anomaly_logger.reset_mock()
        mock_model_client.predict_painting_issue = AsyncMock(return_value={})
        await scheduler_instance._simulate_data_collection()
        mock_anomaly_logger.log_normal_processing.assert_called()

    def test_get_status_scheduler_exception(self, scheduler_instance, mock_settings):
        """Test get_status handles scheduler exceptions gracefully"""
        scheduler_instance.scheduler.get_jobs = Mock(side_effect=Exception("Scheduler error"))
        
        with pytest.raises(Exception, match="Scheduler error"):
            scheduler_instance.get_status()

    @pytest.mark.asyncio
    async def test_datetime_formatting_in_simulation(self, scheduler_instance, mock_azure_storage, mock_model_client, capsys):
        """Test datetime formatting in simulation output"""
        mock_data = {"temperature": 70}
        mock_azure_storage.simulate_real_time_data = AsyncMock(return_value=mock_data)
        mock_model_client.predict_painting_issue = AsyncMock(return_value=None)
        
        with patch('app.services.scheduler_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 15, 30, 45)
            mock_datetime.strftime = datetime.strftime
            
            await scheduler_instance._simulate_data_collection()
            
            captured = capsys.readouterr()
            assert "2024-01-01 15:30:45" in captured.out