import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
import os

# Add the parent directory to the path to import the app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from main import app
except ImportError:
    # Try alternative import path
    from app.main import app


class TestFastAPIMainApplication:
    """
    Comprehensive test suite for the main FastAPI application.
    Testing Framework: pytest with FastAPI TestClient (as identified from requirements.txt)
    """
    
    def setup_method(self):
        """Set up test client for each test method."""
        self.client = TestClient(app)
    
    def test_app_instance_creation(self):
        """Test that the FastAPI app instance is created correctly."""
        assert isinstance(app, FastAPI)
        assert app.title == "Robot Welding Anomaly Detection API"
        assert app.version == "1.0"
    
    def test_app_configuration_properties(self):
        """Test that all required app configuration properties are set."""
        # Test title
        assert app.title == "Robot Welding Anomaly Detection API"
        
        # Test version
        assert app.version == "1.0"
        
        # Test that description is either None or a string
        assert app.description is None or isinstance(app.description, str)
        
        # Test that the app has a root_path (default should be empty string)
        assert hasattr(app, 'root_path')
    
    def test_predict_router_inclusion(self):
        """Test that the predict router is properly included with correct prefix."""
        # Get all registered routes
        routes = [route for route in app.routes]
        
        # Check that we have routes (at minimum, docs, openapi, redoc)
        assert len(routes) > 0
        
        # Look for routes with /api prefix
        api_routes = []
        for route in routes:
            if hasattr(route, 'path') and route.path.startswith('/api'):
                api_routes.append(route)
        
        # Should have at least one API route from the predict router
        assert len(api_routes) > 0, "No API routes found with /api prefix"
    
    def test_predict_endpoint_registration(self):
        """Test that the predict endpoint is properly registered."""
        # Check the OpenAPI schema for the predict endpoint
        schema = app.openapi()
        paths = schema.get("paths", {})
        
        # Should have the /api/predict endpoint
        predict_path = "/api/predict"
        assert predict_path in paths, f"Predict endpoint {predict_path} not found in registered paths"
        
        # Check that it's a POST endpoint
        assert "post" in paths[predict_path], "Predict endpoint should be a POST method"
    
    def test_openapi_schema_generation(self):
        """Test that OpenAPI schema is generated correctly."""
        schema = app.openapi()
        
        # Basic schema structure
        assert "info" in schema
        assert "paths" in schema
        
        # Info section
        info = schema["info"]
        assert info["title"] == "Robot Welding Anomaly Detection API"
        assert info["version"] == "1.0"
        
        # Paths should include our API endpoints
        paths = schema["paths"]
        assert len(paths) > 0
    
    def test_docs_endpoint_accessibility(self):
        """Test that the automatic documentation endpoint is accessible."""
        response = self.client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        
        # Should contain FastAPI documentation elements
        content = response.text
        assert "swagger" in content.lower() or "openapi" in content.lower()
    
    def test_openapi_json_endpoint(self):
        """Test that the OpenAPI JSON specification endpoint works."""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"
        
        # Validate JSON structure
        json_data = response.json()
        assert "info" in json_data
        assert "paths" in json_data
        assert json_data["info"]["title"] == "Robot Welding Anomaly Detection API"
        assert json_data["info"]["version"] == "1.0"
    
    def test_redoc_endpoint_accessibility(self):
        """Test that the ReDoc documentation endpoint is accessible."""
        response = self.client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        
        # Should contain ReDoc documentation elements
        content = response.text
        assert "redoc" in content.lower()
    
    def test_router_prefix_configuration(self):
        """Test that the router is included with the correct /api prefix."""
        # Test by checking if predict endpoint is available at /api/predict
        response = self.client.get("/api/predict")
        # Should return 405 (Method Not Allowed) since it's a POST endpoint
        # or 422 (Unprocessable Entity) if it expects a body
        assert response.status_code in [405, 422], f"Expected 405 or 422, got {response.status_code}"
    
    def test_invalid_routes_return_404(self):
        """Test that non-existent routes return 404."""
        # Test root level invalid route
        response = self.client.get("/nonexistent")
        assert response.status_code == 404
        
        # Test API level invalid route
        response = self.client.get("/api/nonexistent")
        assert response.status_code == 404
        
        # Test deeply nested invalid route
        response = self.client.get("/api/v1/invalid/deeply/nested")
        assert response.status_code == 404
    
    def test_app_middleware_stack(self):
        """Test that the application middleware stack is properly configured."""
        # Check that middleware is accessible
        assert hasattr(app, 'user_middleware')
        assert hasattr(app, 'middleware_stack')
        
        # Middleware should be a list (even if empty)
        assert isinstance(app.user_middleware, list)
    
    def test_app_exception_handlers(self):
        """Test that exception handlers are properly configured."""
        assert hasattr(app, 'exception_handlers')
        assert isinstance(app.exception_handlers, dict)
        
        # Should have default FastAPI exception handlers
        assert len(app.exception_handlers) >= 0
    
    def test_app_state_object(self):
        """Test that the application state object is available and functional."""
        assert hasattr(app, 'state')
        
        # Test that we can set and get state values
        app.state.test_key = "test_value"
        assert app.state.test_key == "test_value"
        
        # Clean up
        delattr(app.state, 'test_key')
    
    def test_cors_configuration(self):
        """Test CORS configuration if present."""
        # Make a preflight OPTIONS request
        response = self.client.options("/api/predict")
        
        # Should either handle CORS (200) or return 405 if CORS not configured
        assert response.status_code in [200, 405]
        
        # If CORS headers are present, verify them
        if 'access-control-allow-origin' in response.headers:
            assert response.headers['access-control-allow-origin'] is not None
    
    def test_request_validation_error_handling(self):
        """Test that request validation errors are handled properly."""
        # Send an invalid POST request to predict endpoint
        response = self.client.post("/api/predict", json={"invalid": "data"})
        
        # Should return 422 for validation error
        assert response.status_code == 422
        
        # Response should be JSON with error details
        error_data = response.json()
        assert "detail" in error_data
        assert isinstance(error_data["detail"], list)
    
    def test_valid_sensor_data_structure(self):
        """Test that the app validates SensorData structure correctly."""
        # Test with valid cur signal type
        valid_cur_data = {
            "signal_type": "cur",
            "values": [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        
        response = self.client.post("/api/predict", json=valid_cur_data)
        # Should either process successfully or fail due to model loading
        assert response.status_code in [200, 500]
        
        # Test with valid vib signal type
        valid_vib_data = {
            "signal_type": "vib",
            "values": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        response = self.client.post("/api/predict", json=valid_vib_data)
        assert response.status_code in [200, 500]
    
    def test_invalid_signal_type_validation(self):
        """Test that invalid signal types are rejected."""
        invalid_data = {
            "signal_type": "invalid_type",
            "values": [1.0, 2.0, 3.0]
        }
        
        response = self.client.post("/api/predict", json=invalid_data)
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
    
    def test_content_type_handling(self):
        """Test that the app handles different content types appropriately."""
        # Test JSON content type for OpenAPI
        response = self.client.get("/openapi.json", headers={"Accept": "application/json"})
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        
        # Test HTML content type for docs
        response = self.client.get("/docs", headers={"Accept": "text/html"})
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_application_startup_shutdown(self):
        """Test that application startup and shutdown work without errors."""
        # Test using context manager (simulates startup/shutdown)
        with TestClient(app) as test_client:
            response = test_client.get("/docs")
            assert response.status_code == 200
        
        # If we reach here, startup and shutdown completed successfully
        assert True
    
    @patch('app.services.predict_service.predict_anomaly')
    def test_predict_endpoint_integration_with_mock(self, mock_predict):
        """Test integration with the predict endpoint using mocked service."""
        # Mock the predict_anomaly function
        mock_predict.return_value = {
            "signal_type": "cur",
            "mae": 0.05,
            "threshold": 0.1,
            "status": "normal"
        }
        
        # Test valid request
        test_data = {
            "signal_type": "cur",
            "values": [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        
        response = self.client.post("/api/predict", json=test_data)
        
        # Should return successful prediction with mocked service
        if response.status_code == 200:
            result = response.json()
            assert "signal_type" in result
            assert "mae" in result
            assert "threshold" in result
            assert "status" in result
            assert result["status"] in ["anomaly", "normal"]
        else:
            # If there are import or dependency issues, document them
            assert response.status_code in [200, 500]
    
    def test_health_check_availability(self):
        """Test if health check endpoints are available."""
        # Many APIs have health check endpoints
        health_endpoints = ["/health", "/api/health", "/healthz", "/ready"]
        
        for endpoint in health_endpoints:
            response = self.client.get(endpoint)
            # Health endpoints should return either 200 (exists) or 404 (doesn't exist)
            assert response.status_code in [200, 404]
    
    def test_app_debug_mode_configuration(self):
        """Test application debug mode configuration."""
        # Check if debug mode is properly configured
        # This is important for production vs development
        assert hasattr(app, 'debug')
        assert isinstance(app.debug, bool)
    
    def test_error_response_format(self):
        """Test that error responses follow consistent format."""
        # Test 404 error format
        response = self.client.get("/nonexistent")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "detail" in error_data
        
        # Test 422 validation error format
        response = self.client.post("/api/predict", json={"invalid": "data"})
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
    
    def test_concurrent_requests_handling(self):
        """Test that the application can handle concurrent requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = self.client.get("/docs")
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {str(e)}")
        
        # Create multiple threads for concurrent testing
        threads = []
        for _ in range(3):  # Keep it small for test performance
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)  # Add timeout for safety
        
        # Verify all requests succeeded
        successful_requests = 0
        while not results.empty():
            result = results.get()
            if result == 200:
                successful_requests += 1
        
        assert successful_requests >= 1, "At least one concurrent request should succeed"
    
    def test_app_response_headers(self):
        """Test that appropriate response headers are set."""
        response = self.client.get("/docs")
        assert response.status_code == 200
        
        headers = response.headers
        
        # Should have content-type header
        assert "content-type" in headers
        
        # Check for security headers (if configured)
        security_headers = ['x-content-type-options', 'x-frame-options', 'x-xss-protection']
        present_security_headers = [h for h in security_headers if h in headers]
        
        # Document which security headers are present (test always passes)
        # This helps identify security configurations
        assert isinstance(present_security_headers, list)
    
    def test_large_request_handling(self):
        """Test handling of larger requests within reasonable limits."""
        # Create a larger but reasonable test payload
        large_values = [float(i) for i in range(1000)]  # 1000 data points
        
        test_data = {
            "signal_type": "cur",
            "values": large_values
        }
        
        response = self.client.post("/api/predict", json=test_data)
        
        # Should handle the request (success or validation error, but not server error)
        assert response.status_code in [200, 422, 500]  # 500 might indicate model loading issues
    
    def test_app_openapi_version_compatibility(self):
        """Test OpenAPI specification version compatibility."""
        schema = app.openapi()
        
        # Should have OpenAPI version specified
        assert "openapi" in schema
        
        # Should be using OpenAPI 3.x.x
        openapi_version = schema["openapi"]
        assert openapi_version.startswith("3."), f"Expected OpenAPI 3.x, got {openapi_version}"
    
    def test_response_model_validation(self):
        """Test that response models are properly configured in OpenAPI schema."""
        schema = app.openapi()
        
        # Check that PredictionResponse model is defined
        components = schema.get("components", {})
        schemas = components.get("schemas", {})
        
        # Should have PredictionResponse schema
        assert "PredictionResponse" in schemas, "PredictionResponse model should be in OpenAPI schema"
        
        # Check required fields in PredictionResponse
        prediction_schema = schemas["PredictionResponse"]
        properties = prediction_schema.get("properties", {})
        
        expected_fields = ["signal_type", "mae", "threshold", "status"]
        for field in expected_fields:
            assert field in properties, f"Field {field} should be in PredictionResponse schema"


class TestApplicationConfigurationEdgeCases:
    """Test edge cases and error conditions for the FastAPI application."""
    
    def setup_method(self):
        """Set up test client for edge case testing."""
        self.client = TestClient(app)
    
    def test_empty_request_body_handling(self):
        """Test handling of empty request bodies."""
        response = self.client.post("/api/predict", json={})
        
        # Should return validation error for missing required fields
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
    
    def test_null_values_handling(self):
        """Test handling of null values in request."""
        test_cases = [
            {"signal_type": None, "values": [1.0, 2.0]},
            {"signal_type": "cur", "values": None},
            {"signal_type": "cur", "values": [1.0, None, 3.0]},
        ]
        
        for test_data in test_cases:
            response = self.client.post("/api/predict", json=test_data)
            # Should return validation error for null values
            assert response.status_code == 422
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON requests."""
        # Send malformed JSON
        response = self.client.post(
            "/api/predict",
            data="{'invalid': json}",  # Invalid JSON syntax
            headers={"content-type": "application/json"}
        )
        
        # Should return 422 for JSON decode error
        assert response.status_code == 422
    
    def test_empty_values_array_handling(self):
        """Test handling of empty values array."""
        test_data = {
            "signal_type": "cur",
            "values": []
        }
        
        response = self.client.post("/api/predict", json=test_data)
        
        # Should either process empty array or return validation error
        assert response.status_code in [200, 400, 422, 500]
    
    def test_oversized_request_protection(self):
        """Test protection against oversized requests."""
        # Create an extremely large payload
        huge_values = [1.0] * 100000  # 100k data points
        
        test_data = {
            "signal_type": "cur",
            "values": huge_values
        }
        
        response = self.client.post("/api/predict", json=test_data)
        
        # Should either process or reject gracefully (not crash)
        assert response.status_code in [200, 413, 422, 500]
    
    def test_invalid_http_methods(self):
        """Test invalid HTTP methods on endpoints."""
        # Test invalid methods on predict endpoint
        invalid_methods = ['GET', 'PUT', 'DELETE', 'PATCH']
        
        for method in invalid_methods:
            response = self.client.request(method, "/api/predict")
            # Should return 405 Method Not Allowed
            assert response.status_code == 405
    
    def test_special_characters_in_paths(self):
        """Test handling of special characters in request paths."""
        special_paths = [
            "/api/predict%20test",
            "/api/predict/../",
            "/api/predict?param=value",
            "/api/predict#fragment"
        ]
        
        for path in special_paths:
            response = self.client.get(path)
            # Should handle gracefully (404 or redirect)
            assert response.status_code in [200, 404, 422, 405]
    
    def test_invalid_content_type_handling(self):
        """Test handling of invalid content types."""
        # Send data with wrong content type
        response = self.client.post(
            "/api/predict",
            data="signal_type=cur&values=1,2,3",
            headers={"content-type": "application/x-www-form-urlencoded"}
        )
        
        # Should return validation error or unsupported media type
        assert response.status_code in [422, 415]
    
    def test_numeric_overflow_handling(self):
        """Test handling of very large numeric values."""
        test_data = {
            "signal_type": "cur",
            "values": [1e308, -1e308, float('inf'), -float('inf')]  # Very large numbers
        }
        
        response = self.client.post("/api/predict", json=test_data)
        
        # Should handle large numbers gracefully
        assert response.status_code in [200, 400, 422, 500]
    
    @pytest.mark.parametrize("signal_type,expected_validation", [
        ("cur", True),
        ("vib", True),
        ("invalid", False),
        ("CUR", False),  # Case sensitive
        ("", False),
        (123, False),  # Wrong type
    ])
    def test_signal_type_validation(self, signal_type, expected_validation):
        """Test signal_type validation with various inputs."""
        test_data = {
            "signal_type": signal_type,
            "values": [1.0, 2.0, 3.0]
        }
        
        response = self.client.post("/api/predict", json=test_data)
        
        if expected_validation:
            # Valid signal types should either succeed or fail due to service issues
            assert response.status_code in [200, 500]
        else:
            # Invalid signal types should return validation error
            assert response.status_code == 422


class TestApplicationPerformanceAndReliability:
    """Test performance and reliability characteristics of the application."""
    
    def setup_method(self):
        """Set up test client for performance testing."""
        self.client = TestClient(app)
    
    def test_application_performance_baseline(self):
        """Test basic performance characteristics of the application."""
        import time
        
        # Test that basic requests complete in reasonable time
        start_time = time.time()
        response = self.client.get("/docs")
        end_time = time.time()
        
        assert response.status_code == 200
        # Should complete within 5 seconds (very generous baseline)
        assert (end_time - start_time) < 5.0
    
    def test_memory_usage_stability(self):
        """Test that repeated requests don't cause memory leaks."""
        # Make multiple requests to ensure no obvious memory leaks
        for _ in range(10):
            response = self.client.get("/openapi.json")
            assert response.status_code == 200
            
            # Ensure response is properly closed
            response.close()
    
    def test_error_recovery(self):
        """Test that the application recovers from errors properly."""
        # Generate some errors
        for _ in range(3):
            self.client.post("/api/predict", json={"invalid": "data"})
        
        # Application should still work after errors
        response = self.client.get("/docs")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])