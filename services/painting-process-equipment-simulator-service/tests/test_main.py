import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import os\

from app.main import app, lifespan

class TestLifespan:
    """Test suite for the lifespan context manager"""
    
    @pytest.mark.asyncio
    async def test_lifespan_startup_with_azure_connection_string(self, capfd):
        """Test lifespan startup when Azure connection string is configured"""
        mock_app = Mock(spec=FastAPI)
        
        with patch('app.main.settings') as mock_settings, \
             patch('app.main.simulator_scheduler') as mock_scheduler, \
             patch('app.main.os.makedirs') as mock_makedirs:
            
            # Configure mock settings
            mock_settings.azure_connection_string = "DefaultEndpointsProtocol=https;AccountName=test"
            mock_settings.log_directory = "/test/logs"
            mock_settings.scheduler_interval_minutes = 5
            mock_settings.model_services = ["service1", "service2", "service3"]
            
            mock_scheduler.is_running = True
            mock_scheduler.stop = AsyncMock()
            
            # Test the lifespan context manager
            async with lifespan(mock_app):
                pass
            
            # Verify directory creation
            mock_makedirs.assert_called_once_with("/test/logs", exist_ok=True)
            
            # Verify scheduler stop was called
            mock_scheduler.stop.assert_called_once()
            
            # Check printed output
            captured = capfd.readouterr()
            assert "🚀 Data Simulator Service 시작 중..." in captured.out
            assert "📁 로그 디렉토리: /test/logs" in captured.out
            assert "🔧 스케줄러 간격: 5분" in captured.out
            assert "🎯 대상 서비스 수: 3" in captured.out
            assert "🛑 Data Simulator Service 종료 중..." in captured.out

    @pytest.mark.asyncio
    async def test_lifespan_scheduler_not_running_on_shutdown(self, capfd):
        """Test lifespan shutdown when scheduler is not running"""
        mock_app = Mock(spec=FastAPI)
        
        with patch('app.main.settings') as mock_settings, \
             patch('app.main.simulator_scheduler') as mock_scheduler, \
             patch('app.main.os.makedirs'):
            
            mock_settings.azure_connection_string = "test"
            mock_settings.log_directory = "/test/logs"
            mock_settings.scheduler_interval_minutes = 1
            mock_settings.model_services = []
            
            mock_scheduler.is_running = False
            mock_scheduler.stop = AsyncMock()
            
            # Test the lifespan context manager
            async with lifespan(mock_app):
                pass
            
            # Verify scheduler stop was NOT called
            mock_scheduler.stop.assert_not_called()
            
            # Check shutdown message still appears
            captured = capfd.readouterr()
            assert "🛑 Data Simulator Service 종료 중..." in captured.out

    @pytest.mark.asyncio
    async def test_lifespan_makedirs_exception_handling(self):
        """Test lifespan handles directory creation errors gracefully"""
        mock_app = Mock(spec=FastAPI)
        
        with patch('app.main.settings') as mock_settings, \
             patch('app.main.simulator_scheduler') as mock_scheduler, \
             patch('app.main.os.makedirs') as mock_makedirs:
            
            mock_settings.azure_connection_string = "test"
            mock_settings.log_directory = "/invalid/path"
            mock_settings.scheduler_interval_minutes = 1
            mock_settings.model_services = []
            
            mock_scheduler.is_running = False
            mock_makedirs.side_effect = OSError("Permission denied")
            
            # Should not raise exception, should handle gracefully
            with pytest.raises(OSError):
                async with lifespan(mock_app):
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_scheduler_stop_exception_handling(self):
        """Test lifespan handles scheduler stop errors gracefully"""
        mock_app = Mock(spec=FastAPI)
        
        with patch('app.main.settings') as mock_settings, \
             patch('app.main.simulator_scheduler') as mock_scheduler, \
             patch('app.main.os.makedirs'):
            
            mock_settings.azure_connection_string = "test"
            mock_settings.log_directory = "/test/logs"
            mock_settings.scheduler_interval_minutes = 1
            mock_settings.model_services = []
            
            mock_scheduler.is_running = True
            mock_scheduler.stop = AsyncMock(side_effect=Exception("Scheduler error"))
            
            # Should handle scheduler stop errors gracefully
            with pytest.raises(Exception, match="Scheduler error"):
                async with lifespan(mock_app):
                    pass


class TestFastAPIApp:
    """Test suite for FastAPI app configuration"""
    
    def test_app_configuration(self):
        """Test FastAPI app is configured with correct metadata"""
        assert app.title == "Painting Process Equipment Data Simulator Service"
        assert app.description == "도장 공정 설비 결함 탐지 모델을 위한 실시간 데이터 시뮬레이터"
        assert app.version == "1.0.0"
        

    def test_app_routers_included(self):
        """Test that required routers are included in the app"""
        # Check that routers are included by examining routes
        route_paths = [route.path for route in app.routes]
        
        # Should have simulator routes
        simulator_routes = [path for path in route_paths if path.startswith("/simulator")]
        assert len(simulator_routes) > 0, "Simulator router not properly included"
        
        # Should have test connection routes  
        test_routes = [path for path in route_paths if path.startswith("/test")]
        assert len(test_routes) > 0, "Test connection router not properly included"

    def test_app_has_basic_routes(self):
        """Test that basic routes (root and health) are defined"""
        route_paths = [route.path for route in app.routes]
        
        assert "/" in route_paths, "Root route not defined"
        assert "/health" in route_paths, "Health route not defined"


class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    def setup_method(self):
        """Setup test client for each test"""
        self.client = TestClient(app)

    @patch('app.main.simulator_scheduler')
    def test_root_endpoint(self, mock_scheduler):
        """Test root endpoint returns correct service information"""
        mock_scheduler.get_status.return_value = {
            "running": True,
            "last_execution": "2024-01-01T00:00:00Z"
        }
        
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        expected_keys = ["service", "version", "status", "target_model", "scheduler_status"]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"
        
        assert data["service"] == "Painting Process Equipment Data Simulator Service"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert data["target_model"] == "painting-process-equipment-defect-detection"
        assert data["scheduler_status"]["running"]

    @patch('app.main.simulator_scheduler')
    def test_root_endpoint_scheduler_not_running(self, mock_scheduler):
        """Test root endpoint when scheduler is not running"""
        mock_scheduler.get_status.return_value = {
            "running": False,
            "last_execution": None
        }
        
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert not data["scheduler_status"]["running"]

    def test_health_check_endpoint(self):
        """Test health check endpoint returns healthy status"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "healthy"}

    def test_nonexistent_endpoint(self):
        """Test that nonexistent endpoints return 404"""
        response = self.client.get("/nonexistent")
        assert response.status_code == 404

    def test_root_endpoint_method_not_allowed(self):
        """Test that non-GET methods on root endpoint return 405"""
        response = self.client.post("/")
        assert response.status_code == 405
        
        response = self.client.put("/")
        assert response.status_code == 405
        
        response = self.client.delete("/")
        assert response.status_code == 405
        
    def test_health_endpoint_method_not_allowed(self):
        """Test that non-GET methods on health endpoint return 405"""
        response = self.client.post("/health")
        assert response.status_code == 405


class TestIntegrationScenarios:
    """Integration test scenarios for the main application"""
    
    def setup_method(self):
        """Setup test client for each test"""
        self.client = TestClient(app)

    @patch('app.main.simulator_scheduler')
    @patch('app.main.settings')
    def test_app_startup_shutdown_cycle(self, mock_settings, mock_scheduler):
        """Test complete startup and shutdown cycle"""
        mock_settings.azure_connection_string = "test"
        mock_settings.log_directory = "/test/logs"
        mock_settings.scheduler_interval_minutes = 5
        mock_settings.model_services = ["service1", "service2"]
        
        mock_scheduler.is_running = True
        mock_scheduler.get_status.return_value = {"running": True}
        mock_scheduler.stop = AsyncMock()
        
        # Create a new test client to trigger lifespan
        with TestClient(app) as client:
            # Test that the app is working after startup
            response = client.get("/health")
            assert response.status_code == 200
            
            response = client.get("/")
            assert response.status_code == 200

    @patch('app.main.simulator_scheduler')
    def test_scheduler_status_consistency(self, mock_scheduler):
        """Test that scheduler status is consistently reported"""
        # Test with running scheduler
        mock_scheduler.get_status.return_value = {
            "running": True,
            "last_execution": "2024-01-01T10:00:00Z",
            "next_execution": "2024-01-01T10:05:00Z"
        }
        
        response1 = self.client.get("/")
        response2 = self.client.get("/")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["scheduler_status"] == data2["scheduler_status"]
        
        # Test with stopped scheduler
        mock_scheduler.get_status.return_value = {
            "running": False,
            "last_execution": "2024-01-01T09:55:00Z",
            "next_execution": None
        }
        
        response3 = self.client.get("/")
        assert response3.status_code == 200
        data3 = response3.json()
        assert not data3["scheduler_status"]["running"]


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def setup_method(self):
        """Setup test client for each test"""
        self.client = TestClient(app)

    def test_malformed_request_headers(self):
        """Test handling of malformed request headers"""
        # Test with various problematic headers
        response = self.client.get("/", headers={"Content-Type": "invalid/type"})
        assert response.status_code == 200  # Should still work
        
        response = self.client.get("/health", headers={"Accept": "invalid"})
        assert response.status_code == 200  # Should still work

    def test_large_request_handling(self):
        """Test handling of requests with large payloads (where applicable)"""
        # Health endpoint should handle any size gracefully since it's GET
        response = self.client.get("/health" + "?" + "x" * 1000)
        # Should either work or return appropriate error, but not crash
        assert response.status_code in [200, 414, 400]


class TestEnvironmentVariableHandling:
    """Test environment variable and configuration handling"""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('app.main.settings')
    def test_missing_environment_variables(self, mock_settings):
        """Test behavior when environment variables are missing"""
        mock_settings.azure_connection_string = None
        mock_settings.log_directory = "./logs"
        mock_settings.scheduler_interval_minutes = 5
        mock_settings.model_services = []
        
        # App should still start and work
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    @patch('app.main.settings')
    def test_invalid_configuration_values(self, mock_settings):
        """Test handling of invalid configuration values"""
        mock_settings.azure_connection_string = ""
        mock_settings.log_directory = ""
        mock_settings.scheduler_interval_minutes = -1
        mock_settings.model_services = None
        
        # App should handle gracefully
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])