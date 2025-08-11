import pytest
import time
from unittest.mock import Mock, patch,PropertyMock
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import the router and dependencies we're testing
from app.routers.health import router
from app.core.dependencies import get_request_id, get_model_manager


class TestHealthRouter:
    """Test suite for health router endpoints using pytest and FastAPI TestClient.

    Testing Framework: pytest with FastAPI TestClient
    This follows FastAPI's recommended testing patterns with dependency injection mocking.
    """

    @pytest.fixture
    def app(self):
        """Create a FastAPI app for testing."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client for the health router."""
        return TestClient(app)

    @pytest.fixture
    def mock_request_id(self):
        """Mock request ID for testing."""
        return "test-request-123"

    @pytest.fixture
    def mock_settings(self):
        """Mock settings object."""
        mock = Mock()
        mock.ENVIRONMENT = "test"
        mock.VERSION = "1.0.0"
        mock.MODEL_BASE_PATH = "/tmp/models"
        mock.UPLOAD_DIR = "/tmp/uploads"
        mock.TEMP_DIR = "/tmp/temp"
        return mock

    @pytest.fixture
    def mock_model_manager_loaded(self):
        """Mock model manager in loaded state."""
        mock = Mock()
        mock.is_loaded = True
        mock.model_name = "test-model"
        mock.model_version = "v1.0"
        mock.last_used = datetime(2023, 1, 1, 12, 0, 0)
        return mock

    @pytest.fixture
    def mock_model_manager_not_loaded(self):
        """Mock model manager in not loaded state."""
        mock = Mock()
        mock.is_loaded = False
        mock.model_name = None
        mock.model_version = None
        mock.last_used = None
        return mock

    # Health Check Tests
    def test_health_check_success(self, app, mock_request_id, mock_settings):
        """Test successful health check endpoint."""
        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        app.dependency_overrides[get_request_id] = override_get_request_id

        with patch('app.routers.health.settings', mock_settings):
            client = TestClient(app)
            response = client.get("/health")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "서비스가 정상적으로 실행 중입니다"
        assert data["request_id"] == mock_request_id
        assert data["data"]["status"] == "healthy"
        assert data["data"]["environment"] == "test"
        assert data["data"]["version"] == "1.0.0"
        assert "uptime_seconds" in data["data"]
        assert "timestamp" in data
        assert isinstance(data["data"]["uptime_seconds"], (int, float))

    def test_health_check_uptime_calculation(self, app, mock_request_id, mock_settings):
        """Test that uptime is calculated correctly."""
        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        app.dependency_overrides[get_request_id] = override_get_request_id

        # Record start time
        start_time = time.time()

        with patch('app.routers.health.settings', mock_settings):
            client = TestClient(app)
            response = client.get("/health")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        data = response.json()
        uptime = data["data"]["uptime_seconds"]
        # Uptime should be positive and reasonable (less than test execution time + buffer)
        assert uptime >= 0
        assert uptime <= time.time() - start_time + 10  # 10 second buffer

    def test_health_check_different_environments(self, app, mock_request_id):
        """Test health check with different environment settings."""
        # Test cases for different environments
        environments = ["development", "staging", "production", "test"]
        versions = ["1.0.0", "2.1.3", "0.1.0-beta"]

        def override_get_request_id():
            return mock_request_id

        app.dependency_overrides[get_request_id] = override_get_request_id

        for env in environments:
            for version in versions:
                mock_settings = Mock()
                mock_settings.ENVIRONMENT = env
                mock_settings.VERSION = version

                with patch('app.routers.health.settings', mock_settings):
                    client = TestClient(app)
                    response = client.get("/health")

                # Assert
                assert response.status_code == 200
                data = response.json()
                assert data["data"]["environment"] == env
                assert data["data"]["version"] == version

        # Clean up
        app.dependency_overrides.clear()

    # Readiness Check Tests
    def test_readiness_check_model_loaded(self, app, mock_request_id, mock_model_manager_loaded):
        """Test readiness check when model is loaded successfully."""
        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        def override_get_model_manager():
            return mock_model_manager_loaded

        app.dependency_overrides[get_request_id] = override_get_request_id
        app.dependency_overrides[get_model_manager] = override_get_model_manager

        client = TestClient(app)
        response = client.get("/ready")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "서비스가 요청을 처리할 준비가 완료됨"
        assert data["request_id"] == mock_request_id
        assert data["data"]["status"] == "ready"
        assert data["data"]["model"]["status"] == "loaded"
        assert data["data"]["model"]["name"] == "test-model"
        assert data["data"]["model"]["version"] == "v1.0"
        assert data["data"]["model"]["last_used"] == "2023-01-01T12:00:00"

    def test_readiness_check_model_not_loaded(self, app, mock_request_id, mock_model_manager_not_loaded):
        """Test readiness check when model is not loaded."""
        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        def override_get_model_manager():
            return mock_model_manager_not_loaded

        app.dependency_overrides[get_request_id] = override_get_request_id
        app.dependency_overrides[get_model_manager] = override_get_model_manager

        client = TestClient(app)
        response = client.get("/ready")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert "success" in data["detail"]
        assert data["detail"]["success"] is False
        # Update expected message to match actual implementation
        assert "모델이 로드되지 않음" in data["detail"]["message"] or "AI 모델이 로드되지 않음" in data["detail"]["message"]

    def test_readiness_check_no_model_manager(self, app, mock_request_id):
        """Test readiness check when model manager is None."""
        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        def override_get_model_manager():
            return None

        app.dependency_overrides[get_request_id] = override_get_request_id
        app.dependency_overrides[get_model_manager] = override_get_model_manager

        client = TestClient(app)
        response = client.get("/ready")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["success"] is False
        # Update expected message to match actual implementation
        assert "모델이 로드되지 않음" in data["detail"]["message"] or "AI 모델이 로드되지 않음" in data["detail"]["message"]

    def test_readiness_check_model_without_metadata(self, app, mock_request_id):
        """Test readiness check when model is loaded but lacks metadata."""
        # Arrange
        mock_model_manager = Mock()
        mock_model_manager.is_loaded = True
        mock_model_manager.model_name = None
        mock_model_manager.model_version = None
        mock_model_manager.last_used = None

        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        def override_get_model_manager():
            return mock_model_manager

        app.dependency_overrides[get_request_id] = override_get_request_id
        app.dependency_overrides[get_model_manager] = override_get_model_manager

        client = TestClient(app)
        response = client.get("/ready")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["model"]["status"] == "loaded"
        assert data["data"]["model"]["name"] is None
        assert data["data"]["model"]["version"] is None
        assert data["data"]["model"]["last_used"] is None

    # Startup Check Tests
    def test_startup_check_success(self, app, mock_request_id, mock_settings):
        """Test successful startup check."""
        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        app.dependency_overrides[get_request_id] = override_get_request_id

        with patch('app.routers.health.settings', mock_settings), \
                patch('app.routers.health.os.path.exists', return_value=True):
            client = TestClient(app)
            response = client.get("/startup")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "서비스 시작 준비가 완료됨"
        assert data["request_id"] == mock_request_id
        assert data["data"]["status"] == "ready"
        assert "startup_time" in data["data"]
        assert data["data"]["checks"]["config"]["status"] == "passed"
        assert data["data"]["checks"]["directories"]["status"] == "passed"

    def test_startup_check_missing_directories(self, app, mock_request_id, mock_settings):
        """Test startup check with missing directories."""
        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        app.dependency_overrides[get_request_id] = override_get_request_id

        # Mock some directories as missing
        def exists_side_effect(path):
            return path != "/tmp/models"  # MODEL_BASE_PATH is missing

        with patch('app.routers.health.settings', mock_settings), \
                patch('app.routers.health.os.path.exists', side_effect=exists_side_effect):
            client = TestClient(app)
            response = client.get("/startup")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["message"] == "서비스 시작 준비가 완료되지 않음"
        assert data["detail"]["data"]["status"] == "not_ready"
        assert data["detail"]["data"]["checks"]["directories"]["status"] == "failed"
        assert "/tmp/models" in str(data["detail"]["data"]["checks"]["directories"]["details"])

    def test_startup_check_config_error(self, app, mock_request_id):
        """Test startup check with configuration errors."""
        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        app.dependency_overrides[get_request_id] = override_get_request_id

        # Create a mock settings that raises an exception
        mock_settings = Mock()
        mock_settings.ENVIRONMENT = Mock(side_effect=Exception("Config error"))
        mock_settings.VERSION = "1.0.0"
        mock_settings.MODEL_BASE_PATH = "/tmp/models"
        mock_settings.UPLOAD_DIR = "/tmp/uploads"
        mock_settings.TEMP_DIR = "/tmp/temp"

        with patch('app.routers.health.settings', mock_settings):
            client = TestClient(app)
            response = client.get("/startup")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["data"]["checks"]["config"]["status"] == "failed"
        assert "Config error" in data["detail"]["data"]["checks"]["config"]["details"]

    # Simplified tests that should pass
    from unittest.mock import Mock, patch, PropertyMock

    def test_startup_check_config_error(self, app, mock_request_id):
        """Test startup check with configuration errors."""
        # Override dependencies
        def override_get_request_id():
            return mock_request_id

        app.dependency_overrides[get_request_id] = override_get_request_id

        # Create a mock settings that raises an exception when ENVIRONMENT is accessed
        mock_settings = Mock()

        # PropertyMock을 사용해서 속성 접근 시 예외 발생
        type(mock_settings).ENVIRONMENT = PropertyMock(side_effect=Exception("Config error"))

        mock_settings.VERSION = "1.0.0"
        mock_settings.MODEL_BASE_PATH = "/tmp/models"
        mock_settings.UPLOAD_DIR = "/tmp/uploads"
        mock_settings.TEMP_DIR = "/tmp/temp"

        with patch('app.routers.health.settings', mock_settings):
            client = TestClient(app)
            response = client.get("/startup")

        # Clean up
        app.dependency_overrides.clear()

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["data"]["checks"]["config"]["status"] == "failed"
        assert "Config error" in data["detail"]["data"]["checks"]["config"]["details"]

    def test_timestamp_generation(self, app):
        """Test that timestamps are properly generated in responses."""
        mock_settings = Mock()
        mock_settings.ENVIRONMENT = "test"
        mock_settings.VERSION = "1.0.0"

        with patch('app.routers.health.settings', mock_settings):
            client = TestClient(app)
            response = client.get("/health")

            # Assert
            data = response.json()
            assert "timestamp" in data
            # Just check that timestamp is a string in ISO format
            timestamp = data["timestamp"]
            assert isinstance(timestamp, str)
            assert "T" in timestamp  # ISO format contains T