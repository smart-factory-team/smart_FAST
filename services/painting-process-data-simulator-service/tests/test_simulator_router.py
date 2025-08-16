import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_scheduler():
    """Mock the simulator_scheduler service."""
    with patch('app.routers.simulator_router.simulator_scheduler', autospec=True) as mock:
        mock.get_status.return_value = {"is_running": False, "next_run": None}
        mock.start = AsyncMock()
        mock.stop = AsyncMock()
        yield mock

def test_get_simulator_status(mock_scheduler):
    """Test GET /simulator/status endpoint."""
    response = client.get("/simulator/status")
    assert response.status_code == 200
    assert response.json() == {"is_running": False, "next_run": None}
    mock_scheduler.get_status.assert_called_once()

def test_start_simulator(mock_scheduler):
    """Test POST /simulator/start endpoint."""
    response = client.post("/simulator/start")
    assert response.status_code == 200
    assert "시뮬레이터가 시작되었습니다." in response.json()["message"]
    mock_scheduler.start.assert_awaited_once()

def test_stop_simulator(mock_scheduler):
    """Test POST /simulator/stop endpoint."""
    response = client.post("/simulator/stop")
    assert response.status_code == 200
    assert "시뮬레이터가 중지되었습니다." in response.json()["message"]
    mock_scheduler.stop.assert_awaited_once()

def test_start_simulator_exception(mock_scheduler):
    """Test exception handling when starting the simulator."""
    mock_scheduler.start.side_effect = Exception("Test Exception")
    response = client.post("/simulator/start")
    assert response.status_code == 500
    assert "시뮬레이터 시작 실패" in response.json()["detail"]
