import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
import sys
import os

# Add the app directory to Python path for imports - following project pattern
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from app.services.test_scheduler_service import SimulatorScheduler, simulator_scheduler


class TestSimulatorSchedulerInitialization:
    """
    Test suite for SimulatorScheduler initialization and basic properties.
    Testing Framework: pytest (as identified from requirements-dev.txt)
    """
    
    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler instance for each test"""
        return SimulatorScheduler()
    
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization with default values"""
        assert scheduler.scheduler is not None
        assert scheduler.is_running is False
        assert hasattr(scheduler.scheduler, 'add_job')
        assert hasattr(scheduler.scheduler, 'start')
        assert hasattr(scheduler.scheduler, 'shutdown')
    
    def test_scheduler_is_asyncio_scheduler(self, scheduler):
        """Test that scheduler uses AsyncIOScheduler"""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        assert isinstance(scheduler.scheduler, AsyncIOScheduler)
    
    def test_global_scheduler_instance(self):
        """Test that global scheduler instance is properly created"""
        assert simulator_scheduler is not None
        assert isinstance(simulator_scheduler, SimulatorScheduler)
        assert hasattr(simulator_scheduler, 'scheduler')
        assert hasattr(simulator_scheduler, 'is_running')


class TestSimulatorSchedulerStart:
    """Test suite for scheduler start functionality"""
    
    @pytest.fixture
    def scheduler(self):
        return SimulatorScheduler()
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings configuration"""
        with patch('app.services.test_scheduler_service.settings') as mock:
            mock.scheduler_interval_minutes = 5
            mock.model_services = {
                'service1': {'url': 'http://service1'},
                'service2': {'url': 'http://service2'}
            }
            yield mock
    
    @pytest.fixture
    def mock_azure_storage(self):
        """Mock Azure Storage service"""
        with patch('app.services.test_scheduler_service.azure_storage') as mock:
            mock.connect = AsyncMock()
            mock.disconnect = AsyncMock()
            mock.simulate_real_time_data = AsyncMock()
            yield mock
    
    @pytest.fixture
    def mock_model_client(self):
        """Mock model client service"""
        with patch('app.services.test_scheduler_service.model_client') as mock:
            mock.health_check_all = AsyncMock()
            mock.predict_welding_data = AsyncMock()
            yield mock
    
    @pytest.mark.asyncio
    async def test_start_scheduler_success(self, scheduler, mock_settings, mock_azure_storage, mock_model_client):
        """Test successful scheduler start"""
        # Setup mocks
        mock_model_client.health_check_all.return_value = {
            'service1': True,
            'service2': True
        }
        
        with patch.object(scheduler.scheduler, 'add_job') as mock_add_job, \
             patch.object(scheduler.scheduler, 'start') as mock_start:
            
            await scheduler.start()
            
            # Verify Azure storage connection
            mock_azure_storage.connect.assert_called_once()
            
            # Verify health check
            mock_model_client.health_check_all.assert_called_once()
            
            # Verify job scheduling
            mock_add_job.assert_called_once()
            job_call = mock_add_job.call_args
            assert job_call.kwargs['id'] == 'data_simulation'
            assert job_call.kwargs['name'] == 'Data Collection Simulation'
            assert job_call.kwargs['replace_existing'] is True
            
            # Verify scheduler start
            mock_start.assert_called_once()
            assert scheduler.is_running is True
    
    @pytest.mark.asyncio
    async def test_start_scheduler_already_running(self, scheduler, mock_settings, mock_azure_storage, mock_model_client, capsys):
        """Test starting scheduler when already running"""
        scheduler.is_running = True
        
        await scheduler.start()
        
        captured = capsys.readouterr()
        assert "⚠️ 스케줄러가 이미 실행 중입니다." in captured.out
        mock_azure_storage.connect.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_start_scheduler_all_services_unhealthy(self, scheduler, mock_settings, mock_azure_storage, mock_model_client):
        """Test scheduler start failure when all services are unhealthy"""
        mock_model_client.health_check_all.return_value = {
            'service1': False,
            'service2': False
        }
        
        with pytest.raises(Exception, match="모든 모델 서비스가 비활성 상태입니다."):
            await scheduler.start()
        
        assert scheduler.is_running is False
    
    @pytest.mark.asyncio
    async def test_start_scheduler_partial_services_healthy(self, scheduler, mock_settings, mock_azure_storage, mock_model_client):
        """Test scheduler start with partially healthy services"""
        mock_model_client.health_check_all.return_value = {
            'service1': True,
            'service2': False
        }
        
        with patch.object(scheduler.scheduler, 'add_job'), \
             patch.object(scheduler.scheduler, 'start'):
            
            await scheduler.start()
            
            assert scheduler.is_running is True
    
    @pytest.mark.asyncio
    async def test_start_scheduler_azure_connection_failure(self, scheduler, mock_settings, mock_azure_storage, mock_model_client):
        """Test scheduler start failure during Azure connection"""
        mock_azure_storage.connect.side_effect = Exception("Azure connection failed")
        
        with pytest.raises(Exception, match="Azure connection failed"):
            await scheduler.start()
        
        assert scheduler.is_running is False
    
    @pytest.mark.asyncio
    async def test_concurrent_start_calls(self, scheduler, mock_settings, mock_azure_storage, mock_model_client):
        """Test multiple concurrent start calls"""
        mock_model_client.health_check_all.return_value = {'service1': True}
        
        with patch.object(scheduler.scheduler, 'add_job'), \
             patch.object(scheduler.scheduler, 'start'):
            
            # First call should succeed
            await scheduler.start()
            assert scheduler.is_running is True
            
            # Second call should be ignored
            await scheduler.start()
            
            # Azure connect should only be called once
            assert mock_azure_storage.connect.call_count == 1


class TestSimulatorSchedulerStop:
    """Test suite for scheduler stop functionality"""
    
    @pytest.fixture
    def scheduler(self):
        return SimulatorScheduler()
    
    @pytest.fixture
    def mock_azure_storage(self):
        """Mock Azure Storage service"""
        with patch('app.services.test_scheduler_service.azure_storage') as mock:
            mock.connect = AsyncMock()
            mock.disconnect = AsyncMock()
            yield mock
    
    @pytest.mark.asyncio
    async def test_stop_scheduler_success(self, scheduler, mock_azure_storage):
        """Test successful scheduler stop"""
        scheduler.is_running = True
        
        with patch.object(scheduler.scheduler, 'shutdown') as mock_shutdown:
            await scheduler.stop()
            
            mock_shutdown.assert_called_once()
            mock_azure_storage.disconnect.assert_called_once()
            assert scheduler.is_running is False
    
    @pytest.mark.asyncio
    async def test_stop_scheduler_not_running(self, scheduler, mock_azure_storage, capsys):
        """Test stopping scheduler when not running"""
        scheduler.is_running = False
        
        await scheduler.stop()
        
        captured = capsys.readouterr()
        assert "⚠️ 스케줄러가 실행 중이 아닙니다." in captured.out
        mock_azure_storage.disconnect.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_on_stop(self, scheduler, mock_azure_storage):
        """Test that resources are properly cleaned up on stop"""
        scheduler.is_running = True
        
        with patch.object(scheduler.scheduler, 'shutdown') as mock_shutdown:
            await scheduler.stop()
            
            # Verify cleanup
            mock_shutdown.assert_called_once()
            mock_azure_storage.disconnect.assert_called_once()
            assert scheduler.is_running is False


class TestSimulatorSchedulerHealthCheck:
    """Test suite for health check functionality"""
    
    @pytest.fixture
    def scheduler(self):
        return SimulatorScheduler()
    
    @pytest.fixture
    def mock_model_client(self):
        """Mock model client service"""
        with patch('app.services.test_scheduler_service.model_client') as mock:
            mock.health_check_all = AsyncMock()
            yield mock
    
    @pytest.mark.asyncio
    async def test_initial_health_check_all_healthy(self, scheduler, mock_model_client, capsys):
        """Test initial health check with all services healthy"""
        mock_model_client.health_check_all.return_value = {
            'service1': True,
            'service2': True,
            'service3': True
        }
        
        await scheduler._initial_health_check()
        
        captured = capsys.readouterr()
        assert "🔍 모델 서비스 헬스 체크 중..." in captured.out
        assert "✅ service1" in captured.out
        assert "✅ service2" in captured.out
        assert "✅ service3" in captured.out
        assert "📈 활성 서비스: 3/3" in captured.out
    
    @pytest.mark.asyncio
    async def test_initial_health_check_mixed_status(self, scheduler, mock_model_client, capsys):
        """Test initial health check with mixed service status"""
        mock_model_client.health_check_all.return_value = {
            'service1': True,
            'service2': False,
            'service3': True
        }
        
        await scheduler._initial_health_check()
        
        captured = capsys.readouterr()
        assert "✅ service1" in captured.out
        assert "❌ service2" in captured.out
        assert "✅ service3" in captured.out
        assert "📈 활성 서비스: 2/3" in captured.out
    
    @pytest.mark.asyncio
    async def test_initial_health_check_no_services(self, scheduler, mock_model_client):
        """Test initial health check with no services"""
        mock_model_client.health_check_all.return_value = {}
        
        with pytest.raises(Exception, match="모든 모델 서비스가 비활성 상태입니다."):
            await scheduler._initial_health_check()
    
    @pytest.mark.asyncio
    async def test_initial_health_check_all_unhealthy(self, scheduler, mock_model_client):
        """Test initial health check with all services unhealthy"""
        mock_model_client.health_check_all.return_value = {
            'service1': False,
            'service2': False,
            'service3': False
        }
        
        with pytest.raises(Exception, match="모든 모델 서비스가 비활성 상태입니다."):
            await scheduler._initial_health_check()


class TestSimulatorDataCollection:
    """Test suite for data collection simulation functionality"""
    
    @pytest.fixture
    def scheduler(self):
        return SimulatorScheduler()
    
    @pytest.fixture
    def mock_azure_storage(self):
        """Mock Azure Storage service"""
        with patch('app.services.test_scheduler_service.azure_storage') as mock:
            mock.simulate_real_time_data = AsyncMock()
            yield mock
    
    @pytest.fixture
    def mock_model_client(self):
        """Mock model client service"""
        with patch('app.services.test_scheduler_service.model_client') as mock:
            mock.predict_welding_data = AsyncMock()
            yield mock
    
    @pytest.fixture
    def mock_anomaly_logger(self):
        """Mock anomaly logger"""
        with patch('app.services.test_scheduler_service.anomaly_logger') as mock:
            mock.log_anomaly = Mock()
            mock.log_normal_processing = Mock()
            mock.log_error = Mock()
            yield mock
    
    @pytest.mark.asyncio
    async def test_simulate_data_collection_success_normal(self, scheduler, mock_azure_storage, mock_model_client, mock_anomaly_logger, capsys):
        """Test successful data collection simulation with normal status"""
        # Setup mock data
        mock_azure_storage.simulate_real_time_data.return_value = {
            "current": {
                "values": [1.2, 1.3, 1.1],
                "timestamp": "2023-01-01T10:00:00Z"
            },
            "vibration": {
                "values": [0.1, 0.2, 0.15],
                "timestamp": "2023-01-01T10:00:00Z"
            }
        }
        
        mock_model_client.predict_welding_data.return_value = {
            "combined": {
                "status": "normal",
                "combined_logic": "All parameters within normal range"
            }
        }
        
        await scheduler._simulate_data_collection()
        
        # Verify calls
        mock_azure_storage.simulate_real_time_data.assert_called_once()
        mock_model_client.predict_welding_data.assert_called_once()
        mock_anomaly_logger.log_normal_processing.assert_called_once()
        mock_anomaly_logger.log_anomaly.assert_not_called()
        
        captured = capsys.readouterr()
        assert "🔄 Welding Machine 데이터 수집 시작" in captured.out
        assert "📊 전류 데이터: 3 포인트" in captured.out
        assert "📊 진동 데이터: 3 포인트" in captured.out
        assert "✅ 정상 상태" in captured.out
    
    @pytest.mark.asyncio
    async def test_simulate_data_collection_success_anomaly(self, scheduler, mock_azure_storage, mock_model_client, mock_anomaly_logger, capsys):
        """Test successful data collection simulation with anomaly detection"""
        # Setup mock data
        simulated_data = {
            "current": {
                "values": [5.2, 5.3, 5.1],
                "timestamp": "2023-01-01T10:00:00Z"
            },
            "vibration": {
                "values": [2.1, 2.2, 2.15],
                "timestamp": "2023-01-01T10:00:00Z"
            }
        }
        
        mock_azure_storage.simulate_real_time_data.return_value = simulated_data
        
        predictions = {
            "combined": {
                "status": "anomaly",
                "combined_logic": "High current and vibration detected"
            },
            "current_prediction": {"status": "anomaly"},
            "vibration_prediction": {"status": "anomaly"}
        }
        
        mock_model_client.predict_welding_data.return_value = predictions
        
        await scheduler._simulate_data_collection()
        
        # Verify anomaly logging with correct parameters
        mock_anomaly_logger.log_anomaly.assert_called_once_with(
            "welding-machine",
            predictions["combined"],
            {
                "current_data": simulated_data["current"],
                "vibration_data": simulated_data["vibration"],
                "detailed_results": predictions
            }
        )
        mock_anomaly_logger.log_normal_processing.assert_not_called()
        
        captured = capsys.readouterr()
        assert "🚨 이상 감지!" in captured.out
    
    @pytest.mark.asyncio
    async def test_simulate_data_collection_no_data(self, scheduler, mock_azure_storage, mock_model_client, capsys):
        """Test data collection when no data is available"""
        mock_azure_storage.simulate_real_time_data.return_value = None
        
        await scheduler._simulate_data_collection()
        
        mock_model_client.predict_welding_data.assert_not_called()
        
        captured = capsys.readouterr()
        assert "⚠️ 수집할 데이터가 없습니다." in captured.out
    
    @pytest.mark.asyncio
    async def test_simulate_data_collection_empty_data(self, scheduler, mock_azure_storage, mock_model_client, capsys):
        """Test data collection with empty data structure"""
        mock_azure_storage.simulate_real_time_data.return_value = {}
        
        await scheduler._simulate_data_collection()
        
        mock_model_client.predict_welding_data.assert_not_called()
        
        captured = capsys.readouterr()
        assert "⚠️ 수집할 데이터가 없습니다." in captured.out
    
    @pytest.mark.asyncio
    async def test_simulate_data_collection_no_predictions(self, scheduler, mock_azure_storage, mock_model_client, capsys):
        """Test data collection when predictions fail"""
        mock_azure_storage.simulate_real_time_data.return_value = {
            "current": {"values": [1.2], "timestamp": "2023-01-01T10:00:00Z"},
            "vibration": {"values": [0.1], "timestamp": "2023-01-01T10:00:00Z"}
        }
        
        mock_model_client.predict_welding_data.return_value = None
        
        await scheduler._simulate_data_collection()
        
        captured = capsys.readouterr()
        assert "❌ 예측 결과를 받을 수 없습니다." in captured.out
    
    @pytest.mark.asyncio
    async def test_simulate_data_collection_empty_predictions(self, scheduler, mock_azure_storage, mock_model_client, capsys):
        """Test data collection with empty predictions"""
        mock_azure_storage.simulate_real_time_data.return_value = {
            "current": {"values": [1.2], "timestamp": "2023-01-01T10:00:00Z"},
            "vibration": {"values": [0.1], "timestamp": "2023-01-01T10:00:00Z"}
        }
        
        mock_model_client.predict_welding_data.return_value = {}
        
        await scheduler._simulate_data_collection()
        
        captured = capsys.readouterr()
        assert "❌ 예측 결과를 받을 수 없습니다." in captured.out
    
    @pytest.mark.asyncio
    async def test_simulate_data_collection_no_combined_result(self, scheduler, mock_azure_storage, mock_model_client, capsys):
        """Test data collection when combined result is missing"""
        mock_azure_storage.simulate_real_time_data.return_value = {
            "current": {"values": [1.2], "timestamp": "2023-01-01T10:00:00Z"},
            "vibration": {"values": [0.1], "timestamp": "2023-01-01T10:00:00Z"}
        }
        
        mock_model_client.predict_welding_data.return_value = {
            "individual_results": {"status": "normal"}
        }
        
        await scheduler._simulate_data_collection()
        
        captured = capsys.readouterr()
        assert "❌ 조합된 예측 결과가 없습니다." in captured.out
    
    @pytest.mark.asyncio
    async def test_simulate_data_collection_exception_handling(self, scheduler, mock_azure_storage, mock_anomaly_logger, capsys):
        """Test exception handling during data collection"""
        mock_azure_storage.simulate_real_time_data.side_effect = Exception("Storage error")
        
        await scheduler._simulate_data_collection()
        
        mock_anomaly_logger.log_error.assert_called_once_with(
            "welding-machine-scheduler", "Storage error"
        )
        
        captured = capsys.readouterr()
        assert "❌ 데이터 수집 중 오류 발생: Storage error" in captured.out
    
    @pytest.mark.asyncio
    async def test_data_collection_with_empty_values_array(self, scheduler, mock_azure_storage, mock_model_client, mock_anomaly_logger):
        """Test data collection with empty values arrays"""
        mock_azure_storage.simulate_real_time_data.return_value = {
            "current": {"values": [], "timestamp": "2023-01-01T10:00:00Z"},
            "vibration": {"values": [], "timestamp": "2023-01-01T10:00:00Z"}
        }
        
        mock_model_client.predict_welding_data.return_value = {
            "combined": {"status": "normal", "combined_logic": "No data to process"}
        }
        
        await scheduler._simulate_data_collection()
        
        # Verify it handles empty arrays gracefully
        mock_model_client.predict_welding_data.assert_called_once()
        mock_anomaly_logger.log_normal_processing.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_prediction_with_None_combined_logic(self, scheduler, mock_azure_storage, mock_model_client, mock_anomaly_logger):
        """Test handling when combined_logic is None"""
        mock_azure_storage.simulate_real_time_data.return_value = {
            "current": {"values": [1.2], "timestamp": "2023-01-01T10:00:00Z"},
            "vibration": {"values": [0.1], "timestamp": "2023-01-01T10:00:00Z"}
        }
        
        mock_model_client.predict_welding_data.return_value = {
            "combined": {
                "status": "normal",
                "combined_logic": None
            }
        }
        
        with patch('builtins.print') as mock_print:
            await scheduler._simulate_data_collection()
            
            # Should handle None combined_logic gracefully
            mock_anomaly_logger.log_normal_processing.assert_called_once()
            
            # Check that print was called with 'N/A' for None combined_logic
            print_calls = [call.args for call in mock_print.call_args_list]
            found_na_call = any("📋 N/A" in str(call) for call in print_calls)
            assert found_na_call


class TestSimulatorSchedulerStatus:
    """Test suite for scheduler status functionality"""
    
    @pytest.fixture
    def scheduler(self):
        return SimulatorScheduler()
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings configuration"""
        with patch('app.services.test_scheduler_service.settings') as mock:
            mock.scheduler_interval_minutes = 5
            mock.model_services = {
                'service1': {'url': 'http://service1'},
                'service2': {'url': 'http://service2'}
            }
            yield mock
    
    def test_get_status_with_running_scheduler(self, scheduler, mock_settings):
        """Test get_status when scheduler is running with jobs"""
        scheduler.is_running = True
        
        # Mock job with next_run_time
        mock_job = Mock()
        mock_job.next_run_time = datetime(2023, 1, 1, 12, 0, 0)
        
        with patch.object(scheduler.scheduler, 'get_jobs', return_value=[mock_job]):
            status = scheduler.get_status()
            
            assert status["is_running"] is True
            assert status["interval_minutes"] == 5
            assert status["next_run"] == "2023-01-01 12:00:00"
            assert status["total_services"] == 2
    
    def test_get_status_with_stopped_scheduler(self, scheduler, mock_settings):
        """Test get_status when scheduler is stopped"""
        scheduler.is_running = False
        
        with patch.object(scheduler.scheduler, 'get_jobs', return_value=[]):
            status = scheduler.get_status()
            
            assert status["is_running"] is False
            assert status["interval_minutes"] == 5
            assert status["next_run"] is None
            assert status["total_services"] == 2
    
    def test_get_status_no_jobs(self, scheduler, mock_settings):
        """Test get_status when no jobs are scheduled"""
        scheduler.is_running = True
        
        with patch.object(scheduler.scheduler, 'get_jobs', return_value=[]):
            status = scheduler.get_status()
            
            assert status["next_run"] is None
    
    def test_scheduler_state_consistency(self, scheduler, mock_settings):
        """Test scheduler state consistency across operations"""
        initial_state = scheduler.is_running
        
        # State should remain consistent
        status1 = scheduler.get_status()
        status2 = scheduler.get_status()
        
        assert status1["is_running"] == status2["is_running"] == initial_state


class TestIntervalTriggerConfiguration:
    """Test scheduler job configuration and trigger setup"""
    
    @pytest.fixture
    def scheduler(self):
        return SimulatorScheduler()
    
    @pytest.mark.asyncio
    async def test_interval_trigger_configuration(self, scheduler):
        """Test that IntervalTrigger is configured correctly"""
        with patch('app.services.test_scheduler_service.settings') as mock_settings, \
             patch('app.services.test_scheduler_service.azure_storage'), \
             patch('app.services.test_scheduler_service.model_client') as mock_client, \
             patch.object(scheduler.scheduler, 'add_job') as mock_add_job, \
             patch.object(scheduler.scheduler, 'start'):
            
            mock_settings.scheduler_interval_minutes = 10
            mock_settings.model_services = {'service1': True}
            mock_client.health_check_all.return_value = {'service1': True}
            
            await scheduler.start()
            
            # Verify IntervalTrigger was configured with correct interval
            mock_add_job.assert_called_once()
            call_args = mock_add_job.call_args
            trigger = call_args.kwargs['trigger']
            
            # Check that trigger is an IntervalTrigger instance
            from apscheduler.triggers.interval import IntervalTrigger
            assert isinstance(trigger, IntervalTrigger)
    
    @pytest.mark.asyncio
    async def test_job_replacement_existing_true(self, scheduler):
        """Test that replace_existing is set to True for job scheduling"""
        with patch('app.services.test_scheduler_service.settings') as mock_settings, \
             patch('app.services.test_scheduler_service.azure_storage'), \
             patch('app.services.test_scheduler_service.model_client') as mock_client, \
             patch.object(scheduler.scheduler, 'add_job') as mock_add_job, \
             patch.object(scheduler.scheduler, 'start'):
            
            mock_settings.scheduler_interval_minutes = 5
            mock_settings.model_services = {'service1': True}
            mock_client.health_check_all.return_value = {'service1': True}
            
            await scheduler.start()
            
            mock_add_job.assert_called_once()
            call_args = mock_add_job.call_args
            assert call_args.kwargs['replace_existing'] is True
    
    @pytest.mark.asyncio
    async def test_scheduled_function_reference(self, scheduler):
        """Test that the correct function is scheduled"""
        with patch('app.services.test_scheduler_service.settings') as mock_settings, \
             patch('app.services.test_scheduler_service.azure_storage'), \
             patch('app.services.test_scheduler_service.model_client') as mock_client, \
             patch.object(scheduler.scheduler, 'add_job') as mock_add_job, \
             patch.object(scheduler.scheduler, 'start'):
            
            mock_settings.scheduler_interval_minutes = 5
            mock_settings.model_services = {'service1': True}
            mock_client.health_check_all.return_value = {'service1': True}
            
            await scheduler.start()
            
            mock_add_job.assert_called_once()
            call_args = mock_add_job.call_args
            func = call_args.kwargs['func']
            
            # Verify the scheduled function is the data collection method
            assert func == scheduler._simulate_data_collection
    
    @pytest.mark.asyncio
    async def test_scheduler_job_persistence(self, scheduler):
        """Test that jobs persist after being added"""
        with patch('app.services.test_scheduler_service.settings') as mock_settings, \
             patch('app.services.test_scheduler_service.azure_storage'), \
             patch('app.services.test_scheduler_service.model_client') as mock_client, \
             patch.object(scheduler.scheduler, 'start'):
            
            mock_settings.scheduler_interval_minutes = 5
            mock_settings.model_services = {'service1': True}
            mock_client.health_check_all.return_value = {'service1': True}
            
            # Before start, no jobs
            assert len(scheduler.scheduler.get_jobs()) == 0
            
            await scheduler.start()
            
            # After start, should have one job
            jobs = scheduler.scheduler.get_jobs()
            assert len(jobs) == 1
            assert jobs[0].id == 'data_simulation'
            assert jobs[0].name == 'Data Collection Simulation'


class TestEdgeCasesAndErrorScenarios:
    """Additional tests for edge cases and error scenarios"""
    
    @pytest.fixture
    def scheduler(self):
        return SimulatorScheduler() 
    
    @pytest.mark.asyncio
    async def test_malformed_simulated_data(self, scheduler):
        """Test handling of malformed simulated data"""
        with patch('app.services.test_scheduler_service.azure_storage') as mock_storage, \
             patch('app.services.test_scheduler_service.model_client'), \
             patch('app.services.test_scheduler_service.anomaly_logger') as mock_logger:
            
            # Test missing keys in simulated data
            mock_storage.simulate_real_time_data.return_value = {
                "current": {"values": [1.2]}
                # Missing vibration data
            }
            
            await scheduler._simulate_data_collection()
            
            # Should handle gracefully and log error
            mock_logger.log_error.assert_called()
    
    @pytest.mark.asyncio
    async def test_network_timeout_scenarios(self, scheduler):
        """Test handling of network timeouts"""
        with patch('app.services.test_scheduler_service.azure_storage') as mock_storage, \
             patch('app.services.test_scheduler_service.anomaly_logger') as mock_logger:
            
            # Simulate timeout exception
            mock_storage.simulate_real_time_data.side_effect = asyncio.TimeoutError("Network timeout")
            
            await scheduler._simulate_data_collection()
            
            mock_logger.log_error.assert_called_with(
                "welding-machine-scheduler", "Network timeout"
            )
    
    @pytest.mark.asyncio
    async def test_large_data_sets(self, scheduler):
        """Test handling of large data sets"""
        with patch('app.services.test_scheduler_service.azure_storage') as mock_storage, \
             patch('app.services.test_scheduler_service.model_client') as mock_client, \
             patch('app.services.test_scheduler_service.anomaly_logger') as mock_logger:
            
            # Large data set
            large_values = list(range(10000))
            mock_storage.simulate_real_time_data.return_value = {
                "current": {"values": large_values, "timestamp": "2023-01-01T10:00:00Z"},
                "vibration": {"values": large_values, "timestamp": "2023-01-01T10:00:00Z"}
            }
            
            mock_client.predict_welding_data.return_value = {
                "combined": {"status": "normal", "combined_logic": "Processing large dataset"}
            }
            
            await scheduler._simulate_data_collection()
            
            # Verify it can handle large datasets
            mock_client.predict_welding_data.assert_called_once()
            mock_logger.log_normal_processing.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_complete_start_stop_cycle(self, scheduler):
        """Test complete start-stop cycle"""
        with patch('app.services.test_scheduler_service.settings') as mock_settings, \
             patch('app.services.test_scheduler_service.azure_storage') as mock_storage, \
             patch('app.services.test_scheduler_service.model_client') as mock_client, \
             patch.object(scheduler.scheduler, 'add_job'), \
             patch.object(scheduler.scheduler, 'start'), \
             patch.object(scheduler.scheduler, 'shutdown') as mock_shutdown:
            
            mock_settings.scheduler_interval_minutes = 5
            mock_settings.model_services = {'service1': True}
            mock_client.health_check_all.return_value = {'service1': True}
            
            # Start scheduler
            await scheduler.start()
            assert scheduler.is_running is True
            
            # Stop scheduler
            await scheduler.stop()
            assert scheduler.is_running is False
            
            mock_shutdown.assert_called_once()
            mock_storage.disconnect.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])