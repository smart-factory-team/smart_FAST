import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

# pytest 설정 (pytest.ini와 동일한 효과)
def pytest_configure(config):
    """pytest 설정"""
    config.addinivalue_line(
        "markers", "asyncio: 비동기 테스트"
    )
    config.addinivalue_line(
        "markers", "unit: 단위 테스트"
    )
    config.addinivalue_line(
        "markers", "integration: 통합 테스트"
    )

def pytest_collection_modifyitems(config, items):
    """테스트 아이템 수정"""
    for item in items:
        # 비동기 테스트에 자동으로 asyncio 마커 추가
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


@pytest.fixture
def event_loop():
    """비동기 테스트를 위한 이벤트 루프 fixture"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """테스트용 설정 모킹"""
    # MagicMock을 사용해서 Settings 객체를 모킹
    mock = MagicMock()
    mock.azure_connection_string = "test_connection_string"
    mock.azure_container_name = "test-container"
    mock.painting_data_folder = "test-painting"
    mock.scheduler_interval_minutes = 1
    mock.batch_size = 5
    mock.painting_model_url = "http://test-model:8002"
    mock.model_service_url = "http://test-model:8002"  # 프로퍼티 추가
    mock.log_directory = "test-logs"
    mock.log_filename = "test_log.json"
    mock.error_log_filename = "test_error.json"
    mock.http_timeout = 5
    mock.max_retries = 2
    return mock


@pytest.fixture
def mock_azure_storage():
    """Azure Storage 서비스 모킹"""
    mock_storage = AsyncMock()
    mock_storage.connect = AsyncMock()
    mock_storage.disconnect = AsyncMock()
    mock_storage.list_data_files = AsyncMock(return_value=["test1.jpg", "test2.jpg"])
    mock_storage.read_image_data = AsyncMock(return_value=b"test_image_data")
    mock_storage.simulate_painting_surface_data = AsyncMock(return_value={
        "images": ["test1.jpg", "test2.jpg"],
        "metadata": {"source": "test"}
    })
    return mock_storage


@pytest.fixture
def mock_model_client():
    """모델 클라이언트 모킹"""
    mock_client = AsyncMock()
    mock_client.predict_painting_surface_data = AsyncMock(return_value={
        "status": "normal",
        "defect_count": 0,
        "total_count": 2
    })
    mock_client.health_check = AsyncMock(return_value=True)
    return mock_client


@pytest.fixture
def mock_scheduler():
    """스케줄러 서비스 모킹"""
    mock_scheduler = AsyncMock()
    mock_scheduler.is_running = False
    mock_scheduler.start = AsyncMock()
    mock_scheduler.stop = AsyncMock()
    mock_scheduler.get_status = MagicMock(return_value={
        "running": False,
        "started_at": None,
        "jobs": []
    })
    return mock_scheduler


@pytest.fixture
def test_log_directory():
    """테스트용 로그 디렉토리"""
    test_dir = "test-logs"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # 테스트 후 정리
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
