import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.schemas.input import SensorData, SEQUENCE_LENGTH
from app.schemas.output import PredictionResponse
from app.routers.predict_router import router, predict_fault


client = TestClient(app)


class TestPredictRouter:
    """Test suite for the predict router using pytest and FastAPI TestClient."""

    def test_router_configuration(self):
        """Test that the router is properly configured with correct prefix and tags."""
        assert router.prefix == "/predict"
        assert router.tags == ["prediction"]

    @pytest.fixture
    def valid_sensor_data(self):
        """Fixture providing valid sensor data for testing."""
        # Create data with minimum required length (SEQUENCE_LENGTH = 20)
        data_points = [1.0, 2.0, 3.0, 4.0, 5.0] * 8  # 40 data points each
        return {
            "AI0_Vibration": data_points,
            "AI1_Vibration": data_points,
            "AI2_Current": data_points
        }

    @pytest.fixture
    def short_sensor_data(self):
        """Fixture providing sensor data with insufficient length."""
        # Create data with less than SEQUENCE_LENGTH
        short_data = [1.0, 2.0, 3.0, 4.0, 5.0]  # Only 5 data points
        return {
            "AI0_Vibration": short_data,
            "AI1_Vibration": short_data,
            "AI2_Current": short_data
        }

    @pytest.fixture
    def mismatched_length_sensor_data(self):
        """Fixture providing sensor data with mismatched lengths."""
        return {
            "AI0_Vibration": [1.0] * 25,
            "AI1_Vibration": [2.0] * 20,  # Different length
            "AI2_Current": [3.0] * 30     # Different length
        }

    @pytest.fixture
    def mock_prediction_result_normal(self):
        """Fixture providing mock prediction service result for normal condition."""
        return {
            "prediction": "정상",
            "reconstruction_error": 0.05,
            "is_fault": False,
            "attribute_errors": None
        }

    @pytest.fixture
    def mock_prediction_result_fault(self):
        """Fixture providing mock prediction service result for fault condition."""
        return {
            "prediction": "고장",
            "reconstruction_error": 0.85,
            "is_fault": True,
            "attribute_errors": {
                "AI0_Vibration": 0.45,
                "AI1_Vibration": 0.32,
                "AI2_Current": 0.78
            }
        }

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_success_normal_condition(self, mock_predict, valid_sensor_data, mock_prediction_result_normal):
        """Test successful prediction with normal condition."""
        # Arrange
        mock_predict.return_value = mock_prediction_result_normal

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["prediction"] == "정상"
        assert response_data["reconstruction_error"] == 0.05
        assert not response_data["is_fault"]
        assert response_data["attribute_errors"] is None
        mock_predict.assert_called_once()

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_success_fault_detected(self, mock_predict, valid_sensor_data, mock_prediction_result_fault):
        """Test successful prediction when fault is detected."""
        # Arrange
        mock_predict.return_value = mock_prediction_result_fault

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["prediction"] == "고장"
        assert response_data["reconstruction_error"] == 0.85
        assert response_data["is_fault"]
        assert response_data["attribute_errors"] is not None
        assert "AI0_Vibration" in response_data["attribute_errors"]
        assert "AI1_Vibration" in response_data["attribute_errors"]
        assert "AI2_Current" in response_data["attribute_errors"]
        mock_predict.assert_called_once()

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_runtime_error_model_not_loaded(self, mock_predict, valid_sensor_data):
        """Test handling of RuntimeError when model is not loaded (503 Service Unavailable)."""
        # Arrange
        mock_predict.side_effect = RuntimeError("모델, 스케일러 또는 임계값이 로드되지 않았습니다. 서버 로그를 확인하세요.")

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 503
        response_data = response.json()
        assert "모델, 스케일러 또는 임계값이 로드되지 않았습니다" in response_data["detail"]
        mock_predict.assert_called_once()

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_runtime_error_different_message(self, mock_predict, valid_sensor_data):
        """Test handling of RuntimeError with different error message."""
        # Arrange
        mock_predict.side_effect = RuntimeError("GPU memory allocation failed")

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 503
        response_data = response.json()
        assert response_data["detail"] == "GPU memory allocation failed"
        mock_predict.assert_called_once()

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_value_error_exception(self, mock_predict, valid_sensor_data):
        """Test handling of ValueError exceptions during prediction (500 Internal Server Error)."""
        # Arrange
        mock_predict.side_effect = ValueError("입력 데이터 길이(10)가 시퀀스 길이 (20)보다 짧아 예측을 수행할 수 없습니다.")

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 500
        response_data = response.json()
        assert "예측 처리 중 오류 발생:" in response_data["detail"]
        assert "입력 데이터 길이" in response_data["detail"]
        mock_predict.assert_called_once()

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_generic_exception(self, mock_predict, valid_sensor_data):
        """Test handling of generic exceptions during prediction."""
        # Arrange
        mock_predict.side_effect = Exception("Unexpected error during model inference")

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 500
        response_data = response.json()
        assert "예측 처리 중 오류 발생: Unexpected error during model inference" in response_data["detail"]
        mock_predict.assert_called_once()

    def test_predict_fault_invalid_json_format(self):
        """Test handling of invalid JSON in request body."""
        # Act
        response = client.post("/predict", content="invalid json", headers={"Content-Type": "application/json"})

        # Assert
        assert response.status_code == 422

    def test_predict_fault_missing_required_fields(self):
        """Test handling of missing required fields in sensor data."""
        # Arrange
        incomplete_data = {"AI0_Vibration": [1.0] * 25}  # Missing AI1_Vibration and AI2_Current

        # Act
        response = client.post("/predict", json=incomplete_data)

        # Assert
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data

    def test_predict_fault_insufficient_data_length(self, short_sensor_data):
        """Test handling of sensor data with insufficient length."""
        # Act
        response = client.post("/predict", json=short_sensor_data)

        # Assert
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data
        # Should contain validation error about minimum length requirement

    def test_predict_fault_mismatched_data_lengths(self, mismatched_length_sensor_data):
        """Test handling of sensor data with mismatched lengths between lists."""
        # Act
        response = client.post("/predict", json=mismatched_length_sensor_data)

        # Assert
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data
        # Should contain validation error about equal lengths requirement

    def test_predict_fault_empty_request_body(self):
        """Test handling of empty request body."""
        # Act
        response = client.post("/predict", json={})

        # Assert
        assert response.status_code == 422

    def test_predict_fault_invalid_data_types(self):
        """Test handling of invalid data types in sensor data."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": ["invalid", "string", "data"] + [1.0] * 20,
            "AI1_Vibration": [None] * 23,
            "AI2_Current": [True, False] + [1.0] * 21
        }

        # Act
        response = client.post("/predict", json=invalid_data)

        # Assert
        assert response.status_code == 422

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_boundary_values(self, mock_predict, mock_prediction_result_normal):
        """Test prediction with boundary values for sensor data."""
        # Arrange
        boundary_data = {
            "AI0_Vibration": [-999.99] * SEQUENCE_LENGTH,  # Minimum vibration values
            "AI1_Vibration": [999.99] * SEQUENCE_LENGTH,   # Maximum vibration values
            "AI2_Current": [0.0] * SEQUENCE_LENGTH          # Zero current values
        }
        mock_predict.return_value = mock_prediction_result_normal

        # Act
        response = client.post("/predict", json=boundary_data)

        # Assert
        assert response.status_code == 200
        mock_predict.assert_called_once()

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_extreme_reconstruction_error(self, mock_predict, valid_sensor_data):
        """Test prediction with extreme reconstruction error values."""
        # Arrange
        extreme_error_result = {
            "prediction": "고장",
            "reconstruction_error": 999.999,  # Very high error
            "is_fault": True,
            "attribute_errors": {
                "AI0_Vibration": 333.33,
                "AI1_Vibration": 333.33,
                "AI2_Current": 333.33
            }
        }
        mock_predict.return_value = extreme_error_result

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["reconstruction_error"] == 999.999
        assert response_data["is_fault"]

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_minimal_reconstruction_error(self, mock_predict, valid_sensor_data):
        """Test prediction with minimal reconstruction error values."""
        # Arrange
        minimal_error_result = {
            "prediction": "정상",
            "reconstruction_error": 0.000001,  # Very low error
            "is_fault": False,
            "attribute_errors": None
        }
        mock_predict.return_value = minimal_error_result

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["reconstruction_error"] == 0.000001
        assert not response_data["is_fault"]

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_service_called_with_correct_data_structure(self, mock_predict, valid_sensor_data, mock_prediction_result_normal):
        """Test that the predict service is called with correct SensorData object structure."""
        # Arrange
        mock_predict.return_value = mock_prediction_result_normal

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 200
        mock_predict.assert_called_once()
        called_args = mock_predict.call_args
        assert called_args.kwargs["data"] is not None
        # Verify that the service was called with a SensorData instance with correct attributes
        sensor_data = called_args.kwargs["data"]
        assert hasattr(sensor_data, "AI0_Vibration")
        assert hasattr(sensor_data, "AI1_Vibration")
        assert hasattr(sensor_data, "AI2_Current")

    @pytest.mark.asyncio
    async def test_predict_fault_direct_function_call_success(self, valid_sensor_data, mock_prediction_result_normal):
        """Test direct function call to predict_fault with successful prediction."""
        # Arrange
        sensor_data = SensorData(**valid_sensor_data)

        with patch('app.routers.predict_router.predict') as mock_predict:
            mock_predict.return_value = mock_prediction_result_normal

            # Act
            result = await predict_fault(sensor_data)

            # Assert
            assert isinstance(result, PredictionResponse)
            assert result.prediction == "정상"
            assert result.reconstruction_error == 0.05
            assert not result.is_fault

    @pytest.mark.asyncio
    async def test_predict_fault_direct_function_call_runtime_error(self, valid_sensor_data):
        """Test direct function call to predict_fault with RuntimeError."""
        # Arrange
        sensor_data = SensorData(**valid_sensor_data)

        with patch('app.routers.predict_router.predict') as mock_predict:
            mock_predict.side_effect = RuntimeError("Model initialization failed")

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await predict_fault(sensor_data)

            assert exc_info.value.status_code == 503
            assert exc_info.value.detail == "Model initialization failed"

    @pytest.mark.asyncio
    async def test_predict_fault_direct_function_call_generic_exception(self, valid_sensor_data):
        """Test direct function call to predict_fault with generic exception."""
        # Arrange
        sensor_data = SensorData(**valid_sensor_data)

        with patch('app.routers.predict_router.predict') as mock_predict:
            mock_predict.side_effect = Exception("Unexpected error")

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await predict_fault(sensor_data)

            assert exc_info.value.status_code == 500
            assert "예측 처리 중 오류 발생: Unexpected error" in exc_info.value.detail

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_stress_test_multiple_requests(self, mock_predict, valid_sensor_data, mock_prediction_result_normal):
        """Test handling of multiple concurrent requests."""
        # Arrange
        mock_predict.return_value = mock_prediction_result_normal

        # Act & Assert
        for _i in range(10):
            response = client.post("/predict", json=valid_sensor_data)
            assert response.status_code == 200

        assert mock_predict.call_count == 10

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_response_model_compliance(self, mock_predict, valid_sensor_data):
        """Test that response complies with PredictionResponse model."""
        # Arrange
        complete_result = {
            "prediction": "고장",
            "reconstruction_error": 0.75,
            "is_fault": True,
            "attribute_errors": {
                "AI0_Vibration": 0.25,
                "AI1_Vibration": 0.30,
                "AI2_Current": 0.20
            }
        }
        mock_predict.return_value = complete_result

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 200
        response_data = response.json()

        # Verify all expected fields are present and properly typed
        assert isinstance(response_data["prediction"], str)
        assert isinstance(response_data["reconstruction_error"], float)
        assert isinstance(response_data["is_fault"], bool)
        assert isinstance(response_data["attribute_errors"], dict) or response_data["attribute_errors"] is None

    # def test_router_endpoint_registration(self):
    #     """Test that the predict endpoint is properly registered with the router."""
    #     # Get all routes from the router
    #     routes = [route for route in router.routes]
    #     predict_routes = [route for route in routes if hasattr(route, 'path') and route.path == ""]
    #
    #     # Assert
    #     assert len(predict_routes) > 0
    #     predict_route = predict_routes[0]
    #     assert "POST" in predict_route.methods

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_large_dataset(self, mock_predict, mock_prediction_result_normal):
        """Test prediction with large dataset (many data points)."""
        # Arrange
        large_data_points = [float(i % 100) for i in range(1000)]  # 1000 data points each
        large_sensor_data = {
            "AI0_Vibration": large_data_points,
            "AI1_Vibration": large_data_points,
            "AI2_Current": large_data_points
        }
        mock_predict.return_value = mock_prediction_result_normal

        # Act
        response = client.post("/predict", json=large_sensor_data)

        # Assert
        assert response.status_code == 200
        mock_predict.assert_called_once()

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_exact_minimum_length(self, mock_predict, mock_prediction_result_normal):
        """Test prediction with exact minimum required data length."""
        # Arrange
        exact_minimum_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        mock_predict.return_value = mock_prediction_result_normal

        # Act
        response = client.post("/predict", json=exact_minimum_data)

        # Assert
        assert response.status_code == 200
        mock_predict.assert_called_once()

    @patch('app.routers.predict_router.predict')
    def test_predict_fault_all_attribute_errors_present(self, mock_predict, valid_sensor_data):
        """Test that all three attribute errors are properly returned when fault is detected."""
        # Arrange
        fault_result_with_all_attributes = {
            "prediction": "고장",
            "reconstruction_error": 0.95,
            "is_fault": True,
            "attribute_errors": {
                "AI0_Vibration": 0.45,
                "AI1_Vibration": 0.32,
                "AI2_Current": 0.18
            }
        }
        mock_predict.return_value = fault_result_with_all_attributes

        # Act
        response = client.post("/predict", json=valid_sensor_data)

        # Assert
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["attribute_errors"]) == 3
        assert all(key in response_data["attribute_errors"] for key in ["AI0_Vibration", "AI1_Vibration", "AI2_Current"])
        assert all(isinstance(val, float) for val in response_data["attribute_errors"].values())