import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.routers.predict_router import router, PredictionResponse
from app.schemas.input import SensorData


# Create a test client for the router
from fastapi import FastAPI
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestPredictionResponse:
    """Test suite for the PredictionResponse model."""

    def test_prediction_response_valid_anomaly(self):
        """Test PredictionResponse model with valid anomaly data."""
        response_data = {
            "signal_type": "vib",
            "mae": 0.125,
            "threshold": 0.1,
            "status": "anomaly"
        }
        response = PredictionResponse(**response_data)
        assert response.signal_type == "vib"
        assert response.mae == 0.125
        assert response.threshold == 0.1
        assert response.status == "anomaly"

    def test_prediction_response_valid_normal(self):
        """Test PredictionResponse model with valid normal data."""
        response_data = {
            "signal_type": "cur",
            "mae": 0.05,
            "threshold": 0.1,
            "status": "normal"
        }
        response = PredictionResponse(**response_data)
        assert response.signal_type == "cur"
        assert response.mae == 0.05
        assert response.threshold == 0.1
        assert response.status == "normal"

    def test_prediction_response_invalid_status(self):
        """Test PredictionResponse model rejects invalid status values."""
        with pytest.raises(ValueError):
            PredictionResponse(
                signal_type="vib",
                mae=0.125,
                threshold=0.1,
                status="invalid_status"  # Should only accept "anomaly" or "normal"
            )

    def test_prediction_response_negative_mae(self):
        """Test PredictionResponse model with negative MAE value."""
        response_data = {
            "signal_type": "vib",
            "mae": -0.125,
            "threshold": 0.1,
            "status": "anomaly"
        }
        # Should accept negative values as they might be valid in some contexts
        response = PredictionResponse(**response_data)
        assert response.mae == -0.125

    def test_prediction_response_zero_values(self):
        """Test PredictionResponse model with zero values."""
        response_data = {
            "signal_type": "cur",
            "mae": 0.0,
            "threshold": 0.0,
            "status": "normal"
        }
        response = PredictionResponse(**response_data)
        assert response.mae == 0.0
        assert response.threshold == 0.0

    def test_prediction_response_very_large_values(self):
        """Test PredictionResponse model with very large float values."""
        response_data = {
            "signal_type": "vib",
            "mae": 1e10,
            "threshold": 1e9,
            "status": "anomaly"
        }
        response = PredictionResponse(**response_data)
        assert response.mae == 1e10
        assert response.threshold == 1e9


class TestSensorDataValidation:
    """Test suite for SensorData schema validation."""

    def test_sensor_data_valid_vib_signal(self):
        """Test SensorData accepts valid vibration signal type."""
        data = {"signal_type": "vib", "values": [1.0, 2.0, 3.0]}
        sensor_data = SensorData(**data)
        assert sensor_data.signal_type == "vib"
        assert sensor_data.values == [1.0, 2.0, 3.0]

    def test_sensor_data_valid_cur_signal(self):
        """Test SensorData accepts valid current signal type."""
        data = {"signal_type": "cur", "values": [1.0, 2.0, 3.0]}
        sensor_data = SensorData(**data)
        assert sensor_data.signal_type == "cur"
        assert sensor_data.values == [1.0, 2.0, 3.0]

    def test_sensor_data_invalid_signal_type(self):
        """Test SensorData rejects invalid signal types."""
        with pytest.raises(ValueError):
            SensorData(signal_type="invalid", values=[1.0, 2.0, 3.0])

    def test_sensor_data_empty_values(self):
        """Test SensorData with empty values list."""
        data = {"signal_type": "vib", "values": []}
        with pytest.raises(ValidationError):
            SensorData(**data)

    def test_sensor_data_mixed_numeric_types(self):
        """Test SensorData accepts mixed int and float values."""
        data = {"signal_type": "vib", "values": [1, 2.5, 3]}
        sensor_data = SensorData(**data)
        assert sensor_data.values == [1.0, 2.5, 3.0]  # All converted to float


class TestPredictEndpoint:
    """Test suite for the /predict endpoint."""

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_successful_vib_anomaly_detection(self, mock_predict):
        """Test successful prediction returning anomaly status for vibration."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.15,
            "threshold": 0.1,
            "status": "anomaly"
        }

        # Create sensor data with proper length for vib (512 values)
        vib_values = [1.2 + i * 0.01 for i in range(512)]
        sensor_data = {
            "signal_type": "vib",
            "values": vib_values
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()
        assert data["signal_type"] == "vib"
        assert data["mae"] == 0.15
        assert data["threshold"] == 0.1
        assert data["status"] == "anomaly"

        # Verify the service was called with correct parameters
        mock_predict.assert_called_once_with("vib", sensor_data["values"])

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_successful_cur_normal_detection(self, mock_predict):
        """Test successful prediction returning normal status for current."""
        mock_predict.return_value = {
            "signal_type": "cur",
            "mae": 0.05,
            "threshold": 0.1,
            "status": "normal"
        }

        # Create sensor data with proper length for cur (1024 values)
        cur_values = [20.1 + i * 0.001 for i in range(1024)]
        sensor_data = {
            "signal_type": "cur",
            "values": cur_values
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()
        assert data["signal_type"] == "cur"
        assert data["mae"] == 0.05
        assert data["threshold"] == 0.1
        assert data["status"] == "normal"

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_with_exact_vib_length(self, mock_predict):
        """Test prediction with exact required length for vibration (512)."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.02,
            "threshold": 0.05,
            "status": "normal"
        }

        vib_values = [1.5 + i * 0.01 for i in range(512)]
        sensor_data = {
            "signal_type": "vib",
            "values": vib_values
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        assert len(mock_predict.call_args[0][1]) == 512

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_with_exact_cur_length(self, mock_predict):
        """Test prediction with exact required length for current (1024)."""
        mock_predict.return_value = {
            "signal_type": "cur",
            "mae": 0.02,
            "threshold": 0.05,
            "status": "normal"
        }

        cur_values = [2.5 + i * 0.001 for i in range(1024)]
        sensor_data = {
            "signal_type": "cur",
            "values": cur_values
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        assert len(mock_predict.call_args[0][1]) == 1024

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_with_extreme_values(self, mock_predict):
        """Test prediction with extreme numerical values."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.1,
            "threshold": 0.2,
            "status": "anomaly"
        }

        # Create extreme values with proper length
        extreme_values = [0.0] * 256 + [1e6] * 256
        sensor_data = {
            "signal_type": "vib",
            "values": extreme_values
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        mock_predict.assert_called_once_with("vib", extreme_values)

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_with_floating_point_precision(self, mock_predict):
        """Test prediction with high precision floating point values."""
        mock_predict.return_value = {
            "signal_type": "cur",
            "mae": 0.123456789,
            "threshold": 0.987654321,
            "status": "normal"
        }

        precise_values = [1.123456789 + i * 0.000001 for i in range(1024)]
        sensor_data = {
            "signal_type": "cur",
            "values": precise_values
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()
        assert data["mae"] == 0.123456789
        assert data["threshold"] == 0.987654321

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_with_zero_values(self, mock_predict):
        """Test prediction with all zero values."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.0,
            "threshold": 0.1,
            "status": "normal"
        }

        zero_values = [0.0] * 512
        sensor_data = {
            "signal_type": "vib",
            "values": zero_values
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()
        assert data["mae"] == 0.0


class TestPredictEndpointErrorHandling:
    """Test suite for error handling in the /predict endpoint."""

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_handles_file_not_found_error(self, mock_predict):
        """Test handling of FileNotFoundError from predict service."""
        mock_predict.side_effect = FileNotFoundError(
            "Model file missing for vib")

        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 500
        data = response.json()
        assert "Model files not found" in data["detail"]
        assert "Model file missing for vib" in data["detail"]

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_handles_value_error_invalid_signal_type(self, mock_predict):
        """Test handling of ValueError for invalid signal type from service."""
        mock_predict.side_effect = ValueError(
            "signal_type must be either 'cur' or 'vib'")

        sensor_data = {
            "signal_type": "vib",  # Valid for schema but could fail in service
            "values": [1.0] * 512
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 400
        data = response.json()
        assert "Invalid input data" in data["detail"]
        assert "signal_type must be either 'cur' or 'vib'" in data["detail"]

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_handles_value_error_wrong_length(self, mock_predict):
        """Test handling of ValueError for wrong array length."""
        mock_predict.side_effect = ValueError(
            "values must have exactly 512 elements for signal_type 'vib'")

        sensor_data = {
            "signal_type": "vib",
            "values": [1.0, 2.0, 3.0]  # Wrong length
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 400
        data = response.json()
        assert "Invalid input data" in data["detail"]
        assert "values must have exactly 512 elements" in data["detail"]

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_handles_runtime_error_from_prediction_pipeline(self, mock_predict):
        """Test handling of RuntimeError from prediction pipeline."""
        mock_predict.side_effect = RuntimeError(
            "Prediction pipeline failed: model prediction error")

        sensor_data = {
            "signal_type": "cur",
            "values": [1.0] * 1024
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 500
        data = response.json()
        assert "Prediction failed" in data["detail"]
        assert "Prediction pipeline failed" in data["detail"]

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_handles_generic_exception(self, mock_predict):
        """Test handling of generic Exception from predict service."""
        mock_predict.side_effect = AttributeError(
            "'NoneType' object has no attribute 'transform'")

        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 500
        data = response.json()
        assert "Prediction failed" in data["detail"]
        assert "'NoneType' object has no attribute 'transform'" in data["detail"]

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_handles_multiple_exception_types(self, mock_predict):
        """Test handling of different exception types in sequence."""
        exceptions = [
            (FileNotFoundError("Model not found"), 500, "Model files not found"),
            (ValueError("values must be numeric"), 400, "Invalid input data"),
            (RuntimeError("Pipeline failed"), 500, "Prediction failed"),
            (KeyError("Missing threshold"), 500, "Prediction failed")
        ]

        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        for exception, expected_status, expected_message in exceptions:
            mock_predict.side_effect = exception

            response = client.post("/predict", json=sensor_data)

            assert response.status_code == expected_status
            data = response.json()
            assert expected_message in data["detail"]

            mock_predict.reset_mock()

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_preserves_exception_details(self, mock_predict):
        """Test that exception details are properly preserved in HTTP responses."""
        original_error = "values must have exactly 1024 elements for signal_type 'cur'"
        mock_predict.side_effect = ValueError(original_error)

        sensor_data = {
            "signal_type": "cur",
            "values": [1.0] * 512  # Wrong length for cur
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 400
        data = response.json()
        assert original_error in data["detail"]


class TestPredictEndpointValidation:
    """Test suite for input validation in the /predict endpoint."""

    def test_predict_invalid_signal_type_at_schema_level(self):
        """Test endpoint rejects invalid signal_type at Pydantic schema level."""
        sensor_data = {
            "signal_type": "invalid_type",
            "values": [1.0, 2.0, 3.0]
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_predict_missing_signal_type(self):
        """Test endpoint with missing signal_type field."""
        sensor_data = {
            "values": [1.0, 2.0, 3.0]
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        error_details = str(data["detail"])
        assert "signal_type" in error_details.lower()

    def test_predict_missing_values(self):
        """Test endpoint with missing values field."""
        sensor_data = {
            "signal_type": "vib"
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        error_details = str(data["detail"])
        assert "values" in error_details.lower()

    def test_predict_empty_json(self):
        """Test endpoint with empty JSON payload."""
        response = client.post("/predict", json={})

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_predict_null_signal_type(self):
        """Test endpoint with null signal_type."""
        sensor_data = {
            "signal_type": None,
            "values": [1.0, 2.0, 3.0]
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 422

    def test_predict_null_values(self):
        """Test endpoint with null values field."""
        sensor_data = {
            "signal_type": "vib",
            "values": None
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 422

    def test_predict_empty_values_array(self):
        """Test endpoint with empty values array."""
        sensor_data = {
            "signal_type": "vib",
            "values": []
        }

        response = client.post("/predict", json=sensor_data)

        # Empty array passes Pydantic validation but will fail in service
        assert response.status_code in [400, 422, 500]

    def test_predict_invalid_values_types(self):
        """Test endpoint with invalid data types in values array."""
        sensor_data = {
            "signal_type": "vib",
            "values": ["not", "a", "number"]
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 422

    def test_predict_mixed_values_types(self):
        """Test endpoint with mixed valid/invalid data types in values array."""
        sensor_data = {
            "signal_type": "vib",
            "values": [1.0, "not_a_number", 3]  # String in middle
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 422

    def test_predict_values_with_null_elements(self):
        """Test endpoint with null elements in values array."""
        sensor_data = {
            "signal_type": "vib",
            "values": [1.0, None, 3.0]
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 422

    def test_predict_invalid_json_format(self):
        """Test endpoint with malformed JSON."""
        response = client.post(
            "/predict",
            # Trailing comma
            data='{"signal_type": "vib", "values": [1.0, 2.0,}',
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_predict_additional_fields_ignored(self):
        """Test endpoint ignores additional unexpected fields."""
        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512,
            "unexpected_field": "should_be_ignored",
            "another_field": 123
        }

        response = client.post("/predict", json=sensor_data)

        # Pydantic should ignore extra fields by default
        assert response.status_code in [
            200, 400, 500]  # Not 422 for extra fields


class TestPredictEndpointHTTPMethods:
    """Test suite for HTTP method validation."""

    def test_predict_post_method_allowed(self):
        """Test that POST method is allowed."""
        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        response = client.post("/predict", json=sensor_data)

        # Should not return 405 Method Not Allowed
        assert response.status_code != 405

    def test_predict_get_method_not_allowed(self):
        """Test that GET method is not allowed."""
        response = client.get("/predict")
        assert response.status_code == 405

    def test_predict_put_method_not_allowed(self):
        """Test that PUT method is not allowed."""
        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        response = client.put("/predict", json=sensor_data)
        assert response.status_code == 405

    def test_predict_delete_method_not_allowed(self):
        """Test that DELETE method is not allowed."""
        response = client.delete("/predict")
        assert response.status_code == 405

    def test_predict_patch_method_not_allowed(self):
        """Test that PATCH method is not allowed."""
        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        response = client.patch("/predict", json=sensor_data)
        assert response.status_code == 405


class TestPredictEndpointResponseFormat:
    """Test suite for response format validation."""

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_response_has_required_fields(self, mock_predict):
        """Test that response contains all required fields."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.1,
            "threshold": 0.2,
            "status": "normal"
        }

        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()

        # Validate all required fields are present
        required_fields = ["signal_type", "mae", "threshold", "status"]
        for field in required_fields:
            assert field in data

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_response_field_types(self, mock_predict):
        """Test that response fields have correct data types."""
        mock_predict.return_value = {
            "signal_type": "cur",
            "mae": 0.123,
            "threshold": 0.456,
            "status": "anomaly"
        }

        sensor_data = {
            "signal_type": "cur",
            "values": [1.0] * 1024
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()

        # Validate data types
        assert isinstance(data["signal_type"], str)
        assert isinstance(data["mae"], (int, float))
        assert isinstance(data["threshold"], (int, float))
        assert isinstance(data["status"], str)
        assert data["status"] in ["anomaly", "normal"]

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_response_content_type(self, mock_predict):
        """Test that response has correct content type."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.1,
            "threshold": 0.2,
            "status": "normal"
        }

        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")


class TestPredictServiceIntegration:
    """Test suite for service integration aspects."""

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_service_called_with_correct_parameters(self, mock_predict):
        """Test that predict_anomaly service is called with exact parameters."""
        mock_predict.return_value = {
            "signal_type": "cur",
            "mae": 0.05,
            "threshold": 0.1,
            "status": "normal"
        }

        signal_type = "cur"
        values = [20.1 + i * 0.001 for i in range(1024)]

        sensor_data = {
            "signal_type": signal_type,
            "values": values
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200

        # Verify exact parameters passed to service
        mock_predict.assert_called_once()
        call_args = mock_predict.call_args
        assert call_args[0][0] == signal_type  # First positional argument
        assert call_args[0][1] == values       # Second positional argument

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_service_called_once_per_request(self, mock_predict):
        """Test that service is called exactly once per request."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.1,
            "threshold": 0.2,
            "status": "normal"
        }

        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        mock_predict.assert_called_once()

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_multiple_requests_independent(self, mock_predict):
        """Test that multiple requests are handled independently."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.1,
            "threshold": 0.2,
            "status": "normal"
        }

        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        # Make multiple requests
        responses = []
        for _i in range(3):
            response = client.post("/predict", json=sensor_data)
            responses.append(response)

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

        # Service should be called for each request
        assert mock_predict.call_count == 3

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_service_return_value_mapping(self, mock_predict):
        """Test that service return values are correctly mapped to response."""
        service_return = {
            "signal_type": "cur",
            "mae": 0.987654321,
            "threshold": 1.123456789,
            "status": "anomaly"
        }
        mock_predict.return_value = service_return

        sensor_data = {
            "signal_type": "cur",
            "values": [10.5 + i * 0.001 for i in range(1024)]
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()

        # Verify exact mapping
        assert data["signal_type"] == service_return["signal_type"]
        assert data["mae"] == service_return["mae"]
        assert data["threshold"] == service_return["threshold"]
        assert data["status"] == service_return["status"]


class TestPredictEndpointRealWorldScenarios:
    """Test suite for real-world usage scenarios."""

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_realistic_vib_data_normal(self, mock_predict):
        """Test with realistic vibration sensor data for normal operation."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.023,
            "threshold": 0.05,
            "status": "normal"
        }

        # Simulate realistic vibration data with slight variations
        import math
        vib_data = [math.sin(i * 0.1) * 0.1 + 1.0 for i in range(512)]

        sensor_data = {
            "signal_type": "vib",
            "values": vib_data
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "normal"
        assert data["mae"] < data["threshold"]

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_realistic_cur_data_anomaly(self, mock_predict):
        """Test with realistic current sensor data showing anomaly."""
        mock_predict.return_value = {
            "signal_type": "cur",
            "mae": 0.156,
            "threshold": 0.1,
            "status": "anomaly"
        }

        # Simulate current data with anomalous spikes
        base_current = 2.5
        cur_data = []
        for i in range(1024):
            if i % 100 == 0:  # Add anomalous spikes
                cur_data.append(base_current + 5.0)
            else:
                cur_data.append(base_current + (i % 10) * 0.01)

        sensor_data = {
            "signal_type": "cur",
            "values": cur_data
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "anomaly"
        assert data["mae"] > data["threshold"]

    @patch('app.routers.predict_router.predict_anomaly')
    def test_predict_boundary_mae_values(self, mock_predict):
        """Test prediction with MAE exactly at threshold boundary."""
        mock_predict.return_value = {
            "signal_type": "vib",
            "mae": 0.1,
            "threshold": 0.1,
            "status": "normal"  # Equal values should be normal
        }

        sensor_data = {
            "signal_type": "vib",
            "values": [1.0] * 512
        }

        response = client.post("/predict", json=sensor_data)

        assert response.status_code == 200
        data = response.json()
        assert data["mae"] == data["threshold"]
        assert data["status"] == "normal"
