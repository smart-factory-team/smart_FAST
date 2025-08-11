import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# 테스트 대상 FastAPI 애플리케이션 임포트
from app.main import app

# 테스트 클라이언트 생성
client = TestClient(app)

@pytest.fixture
def mock_scheduler():
    """
    simulator_scheduler 서비스를 모의(mock) 처리하는 픽스처.
    autospec=True를 사용하여 실제 객체의 시그니처(동기/비동기)를 유지합니다.
    """
    # patch의 대상은 해당 객체가 '사용되는' 위치인 'app.routers.simulator_router'로 지정합니다.
    with patch('app.routers.simulator_router.simulator_scheduler', autospec=True) as mock:
        # is_running 속성의 기본값 설정
        mock.is_running = False
        
        # get_status (동기 메서드)의 기본 반환값 설정
        mock.get_status.return_value = {
            "is_running": False,
            "interval_minutes": 1,
            "next_run": None,
            "total_services": 1
        }
        
        # start, stop (비동기 메서드)는 autospec에 의해 자동으로 AsyncMock으로 처리됩니다.
        yield mock

@pytest.fixture
def mock_model_client():
    """
    model_client 서비스를 모의(mock) 처리하는 픽스처.
    """
    with patch('app.routers.simulator_router.model_client', autospec=True) as mock:
        # health_check_all (비동기 메서드)의 반환값 설정
        mock.health_check_all.return_value = {
            "painting-process-equipment": True
        }
        yield mock

def test_get_simulator_status_when_stopped(mock_scheduler):
    """
    Given: 시뮬레이터가 중지된 상태일 때
    When: GET /simulator/status API를 호출하면
    Then: 200 OK와 함께 is_running이 False인 상태 정보를 반환해야 합니다.
    """
    # When
    response = client.get("/simulator/status")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["is_running"] is False
    assert "model_services_health" not in data

def test_start_simulator(mock_scheduler):
    """
    Given: 시뮬레이터가 중지된 상태일 때
    When: POST /simulator/start API를 호출하면
    Then: 200 OK와 함께 성공 메시지를 반환하고, 스케줄러의 start 메서드가 호출되어야 합니다.
    """
    # When
    response = client.post("/simulator/start")

    # Then
    assert response.status_code == 200
    assert response.json()["message"] == "시뮬레이터가 시작되었습니다."
    
    # 스케줄러의 start 메서드가 1번 호출되었는지 검증
    mock_scheduler.start.assert_awaited_once()

def test_stop_simulator(mock_scheduler):
    """
    Given: 시뮬레이터가 실행 중인 상태일 때
    When: POST /simulator/stop API를 호출하면
    Then: 200 OK와 함께 성공 메시지를 반환하고, 스케줄러의 stop 메서드가 호출되어야 합니다.
    """
    # Given: 시뮬레이터를 실행 중인 상태로 설정
    mock_scheduler.is_running = True

    # When
    response = client.post("/simulator/stop")

    # Then
    assert response.status_code == 200
    assert response.json()["message"] == "시뮬레이터가 중지되었습니다."
    
    # 스케줄러의 stop 메서드가 1번 호출되었는지 검증
    mock_scheduler.stop.assert_awaited_once()

def test_get_simulator_status_when_running(mock_scheduler, mock_model_client):
    """
    Given: 시뮬레이터가 실행 중인 상태일 때
    When: GET /simulator/status API를 호출하면
    Then: 200 OK와 함께 is_running이 True이고, 모델 서비스 헬스 체크 결과가 포함된 상태를 반환해야 합니다.
    """
    # Given: 시뮬레이터를 실행 중인 상태로 설정
    mock_scheduler.is_running = True
    mock_scheduler.get_status.return_value = {
        "is_running": True,
        "interval_minutes": 1,
        "next_run": "2025-08-06T13:00:00",
        "total_services": 1
    }

    # When
    response = client.get("/simulator/status")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["is_running"] is True
    assert "model_services_health" in data
    assert data["model_services_health"]["painting-process-equipment"] is True
    
    # model_client의 health_check_all 메서드가 1번 호출되었는지 검증
    mock_model_client.health_check_all.assert_awaited_once()