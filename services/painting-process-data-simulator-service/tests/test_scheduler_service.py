import pytest
from unittest.mock import patch, AsyncMock
from app.services.scheduler_service import SimulatorScheduler
from app.config.settings import settings

@pytest.fixture
def scheduler_instance():
    """Provides a clean instance of SimulatorScheduler for each test."""
    return SimulatorScheduler()

@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler_instance):
    """Test the start and stop methods of the scheduler."""
    with patch.object(scheduler_instance.scheduler, 'add_job') as mock_add_job, \
         patch.object(scheduler_instance.scheduler, 'start') as mock_start, \
         patch.object(scheduler_instance.scheduler, 'shutdown') as mock_shutdown: 
        
        await scheduler_instance.start()
        assert scheduler_instance.is_running is True
        mock_add_job.assert_called_once()
        mock_start.assert_called_once()

        await scheduler_instance.stop()
        assert scheduler_instance.is_running is False
        mock_shutdown.assert_called_once()

@pytest.mark.asyncio
async def test_simulate_and_send_data():
    """Test the _simulate_and_send_data method."""
    scheduler = SimulatorScheduler()
    simulated_data = {"key": "value"}

    with patch('app.services.scheduler_service.azure_storage', new_callable=AsyncMock) as mock_azure, \
         patch('app.services.scheduler_service.backend_client', new_callable=AsyncMock) as mock_backend: 
        
        mock_azure.simulate_real_time_data.return_value = simulated_data
        mock_backend.send_to_backend = AsyncMock()

        await scheduler._simulate_and_send_data()

        mock_azure.simulate_real_time_data.assert_awaited_once()
        mock_backend.send_to_backend.assert_awaited_once_with(simulated_data, settings.backend_service_url)

@pytest.mark.asyncio
async def test_simulate_and_send_data_no_data():
    """Test the _simulate_and_send_data method when no data is available."""
    scheduler = SimulatorScheduler()

    with patch('app.services.scheduler_service.azure_storage', new_callable=AsyncMock) as mock_azure, \
         patch('app.services.scheduler_service.backend_client', new_callable=AsyncMock) as mock_backend: 
        
        mock_azure.simulate_real_time_data.return_value = None
        mock_backend.send_to_backend = AsyncMock()

        await scheduler._simulate_and_send_data()

        mock_azure.simulate_real_time_data.assert_awaited_once()
        mock_backend.send_to_backend.assert_not_awaited()

