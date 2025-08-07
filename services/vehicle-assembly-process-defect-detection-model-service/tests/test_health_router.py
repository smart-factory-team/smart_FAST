import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime
from fastapi.testclient import TestClient

# Import the router and dependencies we're testing
from app.routers.health import router

class TestHealthRouter:
    """Test suite for health router endpoints using pytest and FastAPI TestClient.
    
    Testing Framework: pytest with FastAPI TestClient
    This follows FastAPI's recommended testing patterns with dependency injection mocking.
    """

    @pytest.fixture
    def client(self):
        """Create a test client for the health router."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
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
    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.settings')
    def test_health_check_success(self, mock_settings_patch, mock_get_request_id, client, mock_request_id, mock_settings):
        """Test successful health check endpoint."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        mock_settings_patch.ENVIRONMENT = mock_settings.ENVIRONMENT
        mock_settings_patch.VERSION = mock_settings.VERSION

        # Act
        response = client.get("/health")

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

    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.settings')
    def test_health_check_uptime_calculation(self, mock_settings_patch, mock_get_request_id, client, mock_request_id, mock_settings):
        """Test that uptime is calculated correctly."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        mock_settings_patch.ENVIRONMENT = mock_settings.ENVIRONMENT
        mock_settings_patch.VERSION = mock_settings.VERSION
        
        # Record start time
        start_time = time.time()
        
        # Act
        response = client.get("/health")
        
        # Assert
        data = response.json()
        uptime = data["data"]["uptime_seconds"]
        # Uptime should be positive and reasonable (less than test execution time + buffer)
        assert uptime >= 0
        assert uptime <= time.time() - start_time + 10  # 10 second buffer

    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.settings')
    def test_health_check_different_environments(self, mock_settings_patch, mock_get_request_id, client, mock_request_id):
        """Test health check with different environment settings."""
        # Test cases for different environments
        environments = ["development", "staging", "production", "test"]
        versions = ["1.0.0", "2.1.3", "0.1.0-beta"]
        
        for env in environments:
            for version in versions:
                # Arrange
                mock_get_request_id.return_value = mock_request_id
                mock_settings_patch.ENVIRONMENT = env
                mock_settings_patch.VERSION = version
                
                # Act
                response = client.get("/health")
                
                # Assert
                assert response.status_code == 200
                data = response.json()
                assert data["data"]["environment"] == env
                assert data["data"]["version"] == version

    # Readiness Check Tests
    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.get_model_manager')
    def test_readiness_check_model_loaded(self, mock_get_model_manager, mock_get_request_id, client, mock_request_id, mock_model_manager_loaded):
        """Test readiness check when model is loaded successfully."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        mock_get_model_manager.return_value = mock_model_manager_loaded

        # Act
        response = client.get("/ready")

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

    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.get_model_manager')
    def test_readiness_check_model_not_loaded(self, mock_get_model_manager, mock_get_request_id, client, mock_request_id, mock_model_manager_not_loaded):
        """Test readiness check when model is not loaded."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        mock_get_model_manager.return_value = mock_model_manager_not_loaded

        # Act
        response = client.get("/ready")

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert "success" in data["detail"]
        assert data["detail"]["success"] is False
        assert data["detail"]["message"] == "AI 모델이 로드되지 않음"
        assert "AI 모델이 로드되지 않음" in data["detail"]["errors"]
        assert data["detail"]["request_id"] == mock_request_id

    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.get_model_manager')
    def test_readiness_check_no_model_manager(self, mock_get_model_manager, mock_get_request_id, client, mock_request_id):
        """Test readiness check when model manager is None."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        mock_get_model_manager.return_value = None

        # Act
        response = client.get("/ready")

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["message"] == "AI 모델이 로드되지 않음"

    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.get_model_manager')
    def test_readiness_check_model_without_metadata(self, mock_get_model_manager, mock_get_request_id, client, mock_request_id):
        """Test readiness check when model is loaded but lacks metadata."""
        # Arrange
        mock_model_manager = Mock()
        mock_model_manager.is_loaded = True
        mock_model_manager.model_name = None
        mock_model_manager.model_version = None
        mock_model_manager.last_used = None
        
        mock_get_request_id.return_value = mock_request_id
        mock_get_model_manager.return_value = mock_model_manager

        # Act
        response = client.get("/ready")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["model"]["status"] == "loaded"
        assert data["data"]["model"]["name"] is None
        assert data["data"]["model"]["version"] is None
        assert data["data"]["model"]["last_used"] is None

    # Startup Check Tests
    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.settings')
    @patch('app.routers.health.os.path.exists')
    def test_startup_check_success(self, mock_exists, mock_settings_patch, mock_get_request_id, client, mock_request_id, mock_settings):
        """Test successful startup check."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        mock_settings_patch.ENVIRONMENT = mock_settings.ENVIRONMENT
        mock_settings_patch.VERSION = mock_settings.VERSION
        mock_settings_patch.MODEL_BASE_PATH = mock_settings.MODEL_BASE_PATH
        mock_settings_patch.UPLOAD_DIR = mock_settings.UPLOAD_DIR
        mock_settings_patch.TEMP_DIR = mock_settings.TEMP_DIR
        mock_exists.return_value = True  # All directories exist

        # Act
        response = client.get("/startup")

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

    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.settings')
    @patch('app.routers.health.os.path.exists')
    def test_startup_check_missing_directories(self, mock_exists, mock_settings_patch, mock_get_request_id, client, mock_request_id, mock_settings):
        """Test startup check with missing directories."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        mock_settings_patch.ENVIRONMENT = mock_settings.ENVIRONMENT
        mock_settings_patch.VERSION = mock_settings.VERSION
        mock_settings_patch.MODEL_BASE_PATH = mock_settings.MODEL_BASE_PATH
        mock_settings_patch.UPLOAD_DIR = mock_settings.UPLOAD_DIR
        mock_settings_patch.TEMP_DIR = mock_settings.TEMP_DIR
        
        # Mock some directories as missing
        def exists_side_effect(path):
            return path != "/tmp/models"  # MODEL_BASE_PATH is missing
        
        mock_exists.side_effect = exists_side_effect

        # Act
        response = client.get("/startup")

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["message"] == "서비스 시작 준비가 완료되지 않음"
        assert data["detail"]["data"]["status"] == "not_ready"
        assert data["detail"]["data"]["checks"]["directories"]["status"] == "failed"
        assert "/tmp/models" in data["detail"]["data"]["checks"]["directories"]["details"]

    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.settings')
    def test_startup_check_config_error(self, mock_settings_patch, mock_get_request_id, client, mock_request_id):
        """Test startup check with configuration errors."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        
        # Make settings access raise an exception
        mock_settings_patch.ENVIRONMENT = PropertyMock(side_effect=Exception("Config error"))
        mock_settings_patch.VERSION = "1.0.0"
        mock_settings_patch.MODEL_BASE_PATH = "/tmp/models"
        mock_settings_patch.UPLOAD_DIR = "/tmp/uploads"
        mock_settings_patch.TEMP_DIR = "/tmp/temp"

        # Act
        response = client.get("/startup")

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["data"]["checks"]["config"]["status"] == "failed"
        assert "Config error" in data["detail"]["data"]["checks"]["config"]["details"]

    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.settings')
    @patch('app.routers.health.os.path.exists')
    def test_startup_check_multiple_failures(self, mock_exists, mock_settings_patch, mock_get_request_id, client, mock_request_id):
        """Test startup check with multiple failure conditions."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        
        # Config failure
        mock_settings_patch.ENVIRONMENT = PropertyMock(side_effect=Exception("Config error"))
        mock_settings_patch.VERSION = "1.0.0"
        mock_settings_patch.MODEL_BASE_PATH = "/tmp/models"
        mock_settings_patch.UPLOAD_DIR = "/tmp/uploads"
        mock_settings_patch.TEMP_DIR = "/tmp/temp"
        
        # Directory failure
        mock_exists.return_value = False  # All directories missing

        # Act
        response = client.get("/startup")

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["data"]["checks"]["config"]["status"] == "failed"
        assert data["detail"]["data"]["checks"]["directories"]["status"] == "failed"

    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.settings')
    @patch('app.routers.health.os.path.exists')
    def test_startup_check_partial_directory_failure(self, mock_exists, mock_settings_patch, mock_get_request_id, client, mock_request_id, mock_settings):
        """Test startup check with some directories missing."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        mock_settings_patch.ENVIRONMENT = mock_settings.ENVIRONMENT
        mock_settings_patch.VERSION = mock_settings.VERSION
        mock_settings_patch.MODEL_BASE_PATH = mock_settings.MODEL_BASE_PATH
        mock_settings_patch.UPLOAD_DIR = mock_settings.UPLOAD_DIR
        mock_settings_patch.TEMP_DIR = mock_settings.TEMP_DIR
        
        # Mock specific directories as missing
        def exists_side_effect(path):
            missing_paths = ["/tmp/uploads", "/tmp/temp"]
            return path not in missing_paths
        
        mock_exists.side_effect = exists_side_effect

        # Act
        response = client.get("/startup")

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert "/tmp/uploads" in data["detail"]["data"]["checks"]["directories"]["details"]
        assert "/tmp/temp" in data["detail"]["data"]["checks"]["directories"]["details"]
        assert "/tmp/models" not in data["detail"]["data"]["checks"]["directories"]["details"]

    # Edge Cases and Error Conditions
    @patch('app.routers.health.get_request_id')
    @patch('app.routers.health.get_model_manager')
    def test_readiness_check_exception_in_model_manager(self, mock_get_model_manager, mock_get_request_id, client, mock_request_id):
        """Test readiness check when model manager throws an exception."""
        # Arrange
        mock_get_request_id.return_value = mock_request_id
        mock_model_manager = Mock()
        mock_model_manager.is_loaded = PropertyMock(side_effect=Exception("Model access error"))
        mock_get_model_manager.return_value = mock_model_manager

        # Act & Assert
        with pytest.raises(Exception, match="Model access error"):
            client.get("/ready")

    def test_health_endpoints_response_schemas(self, client):
        """Test that all health endpoints return properly structured responses."""
        # This test verifies response structure without mocking dependencies
        with patch('app.routers.health.get_request_id', return_value="test-id"), patch('app.routers.health.settings') as mock_settings:
            mock_settings.ENVIRONMENT = "test"
            mock_settings.VERSION = "1.0.0"
            
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            
            # Verify required fields exist
            required_fields = ["success", "message", "data", "timestamp", "request_id"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Verify data structure
            assert "status" in data["data"]
            assert "uptime_seconds" in data["data"]
            assert "environment" in data["data"]
            assert "version" in data["data"]

    @patch('app.routers.health.time.time')
    def test_service_start_time_consistency(self, mock_time, client):
        """Test that service start time is consistent across calls."""
        # Arrange
        mock_time.return_value = 1000.0  # Fixed time for testing
        
        with patch('app.routers.health.get_request_id', return_value="test-id"), patch('app.routers.health.settings') as mock_settings:
            mock_settings.ENVIRONMENT = "test"
            mock_settings.VERSION = "1.0.0"
            
            # Act - Make multiple calls
            response1 = client.get("/health")
            response2 = client.get("/health")
            
            # Assert - Uptime should be the same since time is mocked
            data1 = response1.json()
            data2 = response2.json()
            assert data1["data"]["uptime_seconds"] == data2["data"]["uptime_seconds"]

    def test_request_id_propagation(self, client):
        """Test that request_id is properly propagated through all endpoints."""
        test_request_id = "unique-test-request-123"
        
        with patch('app.routers.health.get_request_id', return_value=test_request_id), patch('app.routers.health.settings') as mock_settings:
            mock_settings.ENVIRONMENT = "test"
            mock_settings.VERSION = "1.0.0"
            mock_settings.MODEL_BASE_PATH = "/tmp/models"
            mock_settings.UPLOAD_DIR = "/tmp/uploads"
            mock_settings.TEMP_DIR = "/tmp/temp"
            
            with patch('app.routers.health.get_model_manager') as mock_model_mgr:
                mock_model_mgr.return_value.is_loaded = True
                mock_model_mgr.return_value.model_name = "test"
                mock_model_mgr.return_value.model_version = "v1"
                mock_model_mgr.return_value.last_used = datetime.now()
                
                with patch('app.routers.health.os.path.exists', return_value=True):
                    # Test all endpoints
                    health_response = client.get("/health")
                    ready_response = client.get("/ready")
                    startup_response = client.get("/startup")
                    
                    # Assert request_id is present and correct
                    assert health_response.json()["request_id"] == test_request_id
                    assert ready_response.json()["request_id"] == test_request_id
                    assert startup_response.json()["request_id"] == test_request_id

    @patch('app.routers.health.datetime')
    def test_timestamp_generation(self, mock_datetime, client):
        """Test that timestamps are properly generated in responses."""
        # Arrange
        fixed_datetime = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fixed_datetime
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        with patch('app.routers.health.get_request_id', return_value="test-id"), patch('app.routers.health.settings') as mock_settings:
            mock_settings.ENVIRONMENT = "test"
            mock_settings.VERSION = "1.0.0"
            
            # Act
            response = client.get("/health")
            
            # Assert
            data = response.json()
            assert data["timestamp"] == "2023-01-01T12:00:00"

from unittest.mock import PropertyMock

# Add this at the end of the file for additional integration-style tests
class TestHealthRouterIntegration:
    """Integration tests for health router that test endpoint behavior more holistically."""
    
    @pytest.fixture
    def app_client(self):
        """Create a test client with the full FastAPI app setup."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_health_endpoint_performance(self, app_client):
        """Test that health endpoint responds quickly (performance test)."""
        start_time = time.time()
        
        with patch('app.routers.health.get_request_id', return_value="perf-test"), patch('app.routers.health.settings') as mock_settings:
            mock_settings.ENVIRONMENT = "test"
            mock_settings.VERSION = "1.0.0"
            
            response = app_client.get("/health")
        
        end_time = time.time()
        
        # Health check should be very fast (< 100ms)
        assert (end_time - start_time) < 0.1
        assert response.status_code == 200

    def test_concurrent_health_checks(self, app_client):
        """Test multiple concurrent health check requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            with patch('app.routers.health.get_request_id', return_value="concurrent-test"), patch('app.routers.health.settings') as mock_settings:
                mock_settings.ENVIRONMENT = "test"
                mock_settings.VERSION = "1.0.0"
                
                response = app_client.get("/health")
                results.put(response.status_code)
        
        # Start 10 concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert results.qsize() == 10
        while not results.empty():
            assert results.get() == 200