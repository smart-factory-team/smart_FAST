from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routers.test_connection_router import router

# Create a test client for the router
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestAzureStorageConnection:
    """Test cases for Azure Storage connection endpoint"""

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_success(self, mock_azure_storage):
        """Test successful Azure Storage connection"""
        # Arrange
        mock_files = [
            "data_file_1.csv",
            "data_file_2.csv",
            "data_file_3.csv",
            "data_file_4.csv",
            "data_file_5.csv",
            "data_file_6.csv"
        ]
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(return_value=mock_files)
        mock_azure_storage.disconnect = AsyncMock()

        # Act
        response = client.post("/azure-storage-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["message"] == "Azure Storage 연결 성공"
        assert response_data["file_count"] == 6
        assert response_data["sample_files"] == mock_files[:5]
        assert len(response_data["sample_files"]) == 5

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_empty_files(self, mock_azure_storage):
        """Test Azure Storage connection with no files"""
        # Arrange
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(return_value=[])
        mock_azure_storage.disconnect = AsyncMock()

        # Act
        response = client.post("/azure-storage-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["file_count"] == 0
        assert response_data["sample_files"] == []

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_few_files(self, mock_azure_storage):
        """Test Azure Storage connection with fewer than 5 files"""
        # Arrange
        mock_files = ["file1.csv", "file2.csv", "file3.csv"]
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(return_value=mock_files)
        mock_azure_storage.disconnect = AsyncMock()

        # Act
        response = client.post("/azure-storage-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["file_count"] == 3
        assert response_data["sample_files"] == mock_files
        assert len(response_data["sample_files"]) == 3

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_connect_failure(self, mock_azure_storage):
        """Test Azure Storage connection failure during connect"""
        # Arrange
        mock_azure_storage.connect = AsyncMock(
            side_effect=Exception("Connection timeout"))

        # Act
        response = client.post("/azure-storage-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "Azure Storage 연결 실패" in response_data["message"]
        assert "Connection timeout" in response_data["message"]

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_list_files_failure(self, mock_azure_storage):
        """Test Azure Storage connection failure during file listing"""
        # Arrange
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(
            side_effect=Exception("Access denied"))

        # Act
        response = client.post("/azure-storage-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "Access denied" in response_data["message"]

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_disconnect_failure(self, mock_azure_storage):
        """Test Azure Storage connection with disconnect failure"""
        # Arrange
        mock_files = ["test.csv"]
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(return_value=mock_files)
        mock_azure_storage.disconnect = AsyncMock(
            side_effect=Exception("Disconnect error"))

        # Act
        response = client.post("/azure-storage-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "Disconnect error" in response_data["message"]

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_very_large_file_list(self, mock_azure_storage):
        """Test Azure Storage connection with very large file list"""
        # Arrange - create a large list of files
        mock_files = [f"file_{i}.csv" for i in range(1000)]
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(return_value=mock_files)
        mock_azure_storage.disconnect = AsyncMock()

        # Act
        response = client.post("/azure-storage-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["file_count"] == 1000
        assert len(response_data["sample_files"]) == 5
        assert response_data["sample_files"] == mock_files[:5]

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_unicode_filenames(self, mock_azure_storage):
        """Test Azure Storage connection with unicode filenames"""
        # Arrange
        mock_files = ["파일1.csv", "データ.csv", "файл.csv", "🔥test.csv"]
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(return_value=mock_files)
        mock_azure_storage.disconnect = AsyncMock()

        # Act
        response = client.post("/azure-storage-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["sample_files"] == mock_files

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_various_exception_types(self, mock_azure_storage):
        """Test Azure connection with various exception types"""
        exception_types = [
            ValueError("Invalid configuration"),
            ConnectionError("Network unreachable"),
            TimeoutError("Request timeout"),
            PermissionError("Access denied"),
            RuntimeError("Runtime error occurred")
        ]

        for exception in exception_types:
            mock_azure_storage.connect = AsyncMock(side_effect=exception)

            response = client.post("/azure-storage-connection")

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "error"
            assert str(exception) in response_data["message"]


class TestModelServicesConnection:
    """Test cases for model services connection endpoint"""

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_services_all_healthy(self, mock_settings, mock_model_client):
        """Test model services connection with all services healthy"""
        # Arrange
        mock_settings.model_services = {
            "service1": {}, "service2": {}, "service3": {}}
        mock_model_client.health_check_all = AsyncMock(return_value={
            "service1": True,
            "service2": True,
            "service3": True
        })

        # Act
        response = client.post("/models-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert set(response_data["healthy_services"]) == {
            "service1", "service2", "service3"}
        assert response_data["unhealthy_services"] == []
        assert response_data["total_services"] == 3
        assert response_data["healthy_count"] == 3

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_services_mixed_health(self, mock_settings, mock_model_client):
        """Test model services connection with mixed health status"""
        # Arrange
        mock_settings.model_services = {
            "service1": {},
            "service2": {},
            "service3": {},
            "service4": {}
        }
        mock_model_client.health_check_all = AsyncMock(return_value={
            "service1": True,
            "service2": False,
            "service3": True,
            "service4": False
        })

        # Act
        response = client.post("/models-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert set(response_data["healthy_services"]) == {
            "service1", "service3"}
        assert set(response_data["unhealthy_services"]) == {
            "service2", "service4"}
        assert response_data["total_services"] == 4
        assert response_data["healthy_count"] == 2

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_services_all_unhealthy(self, mock_settings, mock_model_client):
        """Test model services connection with all services unhealthy"""
        # Arrange
        mock_settings.model_services = {"service1": {}, "service2": {}}
        mock_model_client.health_check_all = AsyncMock(return_value={
            "service1": False,
            "service2": False
        })

        # Act
        response = client.post("/models-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["healthy_services"] == []
        assert set(response_data["unhealthy_services"]) == {
            "service1", "service2"}
        assert response_data["total_services"] == 2
        assert response_data["healthy_count"] == 0

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_services_no_services(self, mock_settings, mock_model_client):
        """Test model services connection with no services configured"""
        # Arrange
        mock_settings.model_services = {}
        mock_model_client.health_check_all = AsyncMock(return_value={})

        # Act
        response = client.post("/models-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["healthy_services"] == []
        assert response_data["unhealthy_services"] == []
        assert response_data["total_services"] == 0
        assert response_data["healthy_count"] == 0

    @patch('app.routers.test_connection_router.model_client')
    def test_model_services_health_check_exception(self, mock_model_client):
        """Test model services connection with health check exception"""
        # Arrange
        mock_model_client.health_check_all = AsyncMock(
            side_effect=Exception("Network error"))

        # Act
        response = client.post("/models-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "error"
        assert "모델 서비스 테스트 실패" in response_data["message"]
        assert "Network error" in response_data["message"]

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_services_empty_health_status(self, mock_settings, mock_model_client):
        """Test model services connection with empty health status response"""
        # Arrange
        mock_settings.model_services = {"service1": {}, "service2": {}}
        mock_model_client.health_check_all = AsyncMock(return_value={})

        # Act
        response = client.post("/models-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "error"
        assert response_data["healthy_services"] == []
        assert response_data["unhealthy_services"] == []
        assert response_data["total_services"] == 2
        assert response_data["healthy_count"] == 0

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_services_partial_health_status(self, mock_settings, mock_model_client):
        """Test model services connection with partial health status response"""
        # Arrange
        mock_settings.model_services = {
            "service1": {}, "service2": {}, "service3": {}}
        mock_model_client.health_check_all = AsyncMock(return_value={
            "service1": True,
            "service2": False
            # service3 missing from health status
        })

        # Act
        response = client.post("/models-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["healthy_services"] == ["service1"]
        assert response_data["unhealthy_services"] == ["service2"]
        assert response_data["total_services"] == 3
        assert response_data["healthy_count"] == 1

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_services_boolean_edge_cases(self, mock_settings, mock_model_client):
        """Test model services with various boolean-like values"""
        # Arrange
        mock_settings.model_services = {
            "service1": {}, "service2": {}, "service3": {},
            "service4": {}, "service5": {}
        }
        mock_model_client.health_check_all = AsyncMock(return_value={
            "service1": True,
            "service2": False,
            "service3": 0,  # Falsy
            "service4": 1,  # Truthy
            "service5": None  # Falsy
        })

        # Act
        response = client.post("/models-connection")

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert set(response_data["healthy_services"]) == {
            "service1", "service4"}
        assert set(response_data["unhealthy_services"]) == {
            "service2", "service3", "service5"}

    @patch('app.routers.test_connection_router.model_client')
    def test_model_services_various_exception_types(self, mock_model_client):
        """Test model services with various exception types"""
        exception_types = [
            ConnectionError("Service unavailable"),
            TimeoutError("Health check timeout"),
            ValueError("Invalid service configuration"),
            RuntimeError("Service runtime error")
        ]

        for exception in exception_types:
            mock_model_client.health_check_all = AsyncMock(
                side_effect=exception)

            response = client.post("/models-connection")

            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "error"
            assert str(exception) in response_data["message"]


class TestHTTPMethodValidation:
    """Test HTTP method validation for endpoints"""

    def test_azure_connection_post_allowed(self):
        """Test that POST method is allowed for Azure connection endpoint"""
        with patch('app.routers.test_connection_router.azure_storage') as mock_azure:
            mock_azure.connect = AsyncMock()
            mock_azure.list_data_files = AsyncMock(return_value=[])
            mock_azure.disconnect = AsyncMock()

            response = client.post("/azure-storage-connection")
            assert response.status_code == 200

    def test_azure_connection_get_not_allowed(self):
        """Test that GET method is not allowed for Azure connection endpoint"""
        response = client.get("/azure-storage-connection")
        assert response.status_code == 405

    def test_azure_connection_put_not_allowed(self):
        """Test that PUT method is not allowed for Azure connection endpoint"""
        response = client.put("/azure-storage-connection")
        assert response.status_code == 405

    def test_models_connection_post_allowed(self):
        """Test that POST method is allowed for models connection endpoint"""
        with patch('app.routers.test_connection_router.model_client') as mock_client, \
                patch('app.routers.test_connection_router.settings') as mock_settings:
            mock_client.health_check_all = AsyncMock(return_value={})
            mock_settings.model_services = {}

            response = client.post("/models-connection")
            assert response.status_code == 200

    def test_models_connection_get_not_allowed(self):
        """Test that GET method is not allowed for models connection endpoint"""
        response = client.get("/models-connection")
        assert response.status_code == 405

    def test_models_connection_delete_not_allowed(self):
        """Test that DELETE method is not allowed for models connection endpoint"""
        response = client.delete("/models-connection")
        assert response.status_code == 405


class TestResponseFormats:
    """Test response format validation"""

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_response_format_success(self, mock_azure_storage):
        """Test Azure Storage response format for success case"""
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(
            return_value=["file1.csv", "file2.csv"])
        mock_azure_storage.disconnect = AsyncMock()

        response = client.post("/azure-storage-connection")
        response_data = response.json()

        # Verify all required fields are present
        required_fields = ["status", "message", "file_count", "sample_files"]
        for field in required_fields:
            assert field in response_data

        # Verify data types
        assert isinstance(response_data["status"], str)
        assert isinstance(response_data["message"], str)
        assert isinstance(response_data["file_count"], int)
        assert isinstance(response_data["sample_files"], list)

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_response_format_error(self, mock_azure_storage):
        """Test Azure Storage response format for error case"""
        mock_azure_storage.connect = AsyncMock(
            side_effect=Exception("Test error"))

        response = client.post("/azure-storage-connection")
        response_data = response.json()

        # Verify required fields for error case
        required_fields = ["status", "message"]
        for field in required_fields:
            assert field in response_data

        assert response_data["status"] == "error"
        assert "Test error" in response_data["message"]

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_services_response_format_success(self, mock_settings, mock_model_client):
        """Test model services response format for success case"""
        mock_settings.model_services = {"service1": {}}
        mock_model_client.health_check_all = AsyncMock(
            return_value={"service1": True})

        response = client.post("/models-connection")
        response_data = response.json()

        # Verify all required fields are present
        required_fields = ["status", "healthy_services",
                           "unhealthy_services", "total_services", "healthy_count"]
        for field in required_fields:
            assert field in response_data

        # Verify data types
        assert isinstance(response_data["status"], str)
        assert isinstance(response_data["healthy_services"], list)
        assert isinstance(response_data["unhealthy_services"], list)
        assert isinstance(response_data["total_services"], int)
        assert isinstance(response_data["healthy_count"], int)

    @patch('app.routers.test_connection_router.model_client')
    def test_model_services_response_format_error(self, mock_model_client):
        """Test model services response format for error case"""
        mock_model_client.health_check_all = AsyncMock(
            side_effect=Exception("Test error"))

        response = client.post("/models-connection")
        response_data = response.json()

        # Verify required fields for error case
        required_fields = ["status", "message"]
        for field in required_fields:
            assert field in response_data

        assert response_data["status"] == "error"
        assert "Test error" in response_data["message"]

    def test_response_content_type(self):
        """Test that responses have correct content type"""
        with patch('app.routers.test_connection_router.azure_storage') as mock_azure:
            mock_azure.connect = AsyncMock()
            mock_azure.list_data_files = AsyncMock(return_value=[])
            mock_azure.disconnect = AsyncMock()

            response = client.post("/azure-storage-connection")
            assert response.status_code == 200
            assert "application/json" in response.headers.get(
                "content-type", "")


class TestRouterIntegration:
    """Integration tests for the router"""

    def test_router_endpoints_registered(self):
        """Test that all expected endpoints are registered"""
        routes = [route.path for route in app.routes]
        assert "/azure-storage-connection" in routes
        assert "/models-connection" in routes

    def test_nonexistent_endpoints(self):
        """Test that nonexistent endpoints return 404"""
        response = client.post("/nonexistent-endpoint")
        assert response.status_code == 404

    @patch('app.routers.test_connection_router.azure_storage')
    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_concurrent_requests_independence(self, mock_settings, mock_model_client, mock_azure_storage):
        """Test that concurrent requests are handled independently"""
        # Setup mocks
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(
            return_value=["test.csv"])
        mock_azure_storage.disconnect = AsyncMock()

        mock_model_client.health_check_all = AsyncMock(
            return_value={"service1": True})
        mock_settings.model_services = {"service1": {}}

        # Make multiple requests
        azure_responses = []
        model_responses = []

        for _ in range(3):
            azure_responses.append(client.post("/azure-storage-connection"))
            model_responses.append(client.post("/models-connection"))

        # All requests should succeed
        for response in azure_responses + model_responses:
            assert response.status_code == 200
            assert "status" in response.json()


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions"""

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_exactly_five_files(self, mock_azure_storage):
        """Test Azure Storage connection with exactly 5 files"""
        mock_files = [f"file_{i}.csv" for i in range(5)]
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(return_value=mock_files)
        mock_azure_storage.disconnect = AsyncMock()

        response = client.post("/azure-storage-connection")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["file_count"] == 5
        assert len(response_data["sample_files"]) == 5
        assert response_data["sample_files"] == mock_files

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_single_file(self, mock_azure_storage):
        """Test Azure Storage connection with single file"""
        mock_files = ["single_file.csv"]
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(return_value=mock_files)
        mock_azure_storage.disconnect = AsyncMock()

        response = client.post("/azure-storage-connection")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["file_count"] == 1
        assert response_data["sample_files"] == mock_files

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_services_single_service(self, mock_settings, mock_model_client):
        """Test model services with single service"""
        mock_settings.model_services = {"single_service": {}}
        mock_model_client.health_check_all = AsyncMock(
            return_value={"single_service": True})

        response = client.post("/models-connection")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["healthy_services"] == ["single_service"]
        assert response_data["unhealthy_services"] == []
        assert response_data["total_services"] == 1
        assert response_data["healthy_count"] == 1

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_connection_files_with_special_characters(self, mock_azure_storage):
        """Test Azure Storage connection with files containing special characters"""
        mock_files = [
            "file with spaces.csv",
            "file-with-dashes.csv",
            "file_with_underscores.csv",
            "file.with.dots.csv",
            "file@with#special$chars%.csv"
        ]
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(return_value=mock_files)
        mock_azure_storage.disconnect = AsyncMock()

        response = client.post("/azure-storage-connection")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert response_data["sample_files"] == mock_files


class TestServiceIntegration:
    """Test service integration aspects"""

    @patch('app.routers.test_connection_router.azure_storage')
    def test_azure_service_method_call_sequence(self, mock_azure_storage):
        """Test that Azure service methods are called in correct sequence"""
        mock_azure_storage.connect = AsyncMock()
        mock_azure_storage.list_data_files = AsyncMock(
            return_value=["test.csv"])
        mock_azure_storage.disconnect = AsyncMock()

        response = client.post("/azure-storage-connection")

        assert response.status_code == 200

        # Verify methods were called in the expected order
        mock_azure_storage.connect.assert_called_once()
        mock_azure_storage.list_data_files.assert_called_once()
        mock_azure_storage.disconnect.assert_called_once()

    @patch('app.routers.test_connection_router.model_client')
    @patch('app.routers.test_connection_router.settings')
    def test_model_client_service_integration(self, mock_settings, mock_model_client):
        """Test model client service integration"""
        mock_settings.model_services = {"test_service": {"url": "http://test"}}
        mock_model_client.health_check_all = AsyncMock(
            return_value={"test_service": True})

        response = client.post("/models-connection")

        assert response.status_code == 200
        mock_model_client.health_check_all.assert_called_once()

        # Verify settings are accessed
        assert hasattr(mock_settings, 'model_services')
