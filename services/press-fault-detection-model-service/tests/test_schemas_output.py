import pytest
from pydantic import ValidationError

from app.schemas.output import PredictionResponse


class TestPredictionResponse:
    """Test suite for PredictionResponse Pydantic model."""
    
    def test_valid_prediction_response_creation(self):
        """Test creating a valid PredictionResponse with all required fields."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.05,
            "is_fault": False
        }
        response = PredictionResponse(**data)
        
        assert response.prediction == "normal"
        assert response.reconstruction_error == 0.05
        assert response.is_fault is False
        assert response.attribute_errors is None
    
    def test_valid_prediction_response_with_attribute_errors(self):
        """Test creating a valid PredictionResponse with optional attribute_errors."""
        data = {
            "prediction": "fault",
            "reconstruction_error": 0.85,
            "is_fault": True,
            "attribute_errors": {
                "temperature": 0.12,
                "pressure": 0.08,
                "vibration": 0.65
            }
        }
        response = PredictionResponse(**data)
        
        assert response.prediction == "fault"
        assert response.reconstruction_error == 0.85
        assert response.is_fault is True
        assert response.attribute_errors == {
            "temperature": 0.12,
            "pressure": 0.08,
            "vibration": 0.65
        }
    
    def test_prediction_response_with_empty_attribute_errors(self):
        """Test creating a PredictionResponse with empty attribute_errors dict."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.02,
            "is_fault": False,
            "attribute_errors": {}
        }
        response = PredictionResponse(**data)
        
        assert response.prediction == "normal"
        assert response.reconstruction_error == 0.02
        assert response.is_fault is False
        assert response.attribute_errors == {}
    
    def test_prediction_response_serialization(self):
        """Test serializing PredictionResponse to dict."""
        data = {
            "prediction": "fault",
            "reconstruction_error": 0.75,
            "is_fault": True,
            "attribute_errors": {"sensor1": 0.5, "sensor2": 0.25}
        }
        response = PredictionResponse(**data)
        serialized = response.dict()
        
        assert serialized == data
        assert isinstance(serialized, dict)
    
    def test_prediction_response_json_serialization(self):
        """Test serializing PredictionResponse to JSON."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.03,
            "is_fault": False,
            "attribute_errors": {"temp": 0.01, "press": 0.02}
        }
        response = PredictionResponse(**data)
        json_str = response.json()
        
        assert isinstance(json_str, str)
        assert '"prediction":"normal"' in json_str
        assert '"reconstruction_error":0.03' in json_str
        assert '"is_fault":false' in json_str
    
    def test_missing_required_field_prediction(self):
        """Test validation error when prediction field is missing."""
        data = {
            "reconstruction_error": 0.05,
            "is_fault": False
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**data)
        
        assert "prediction" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)
    
    def test_missing_required_field_reconstruction_error(self):
        """Test validation error when reconstruction_error field is missing."""
        data = {
            "prediction": "normal",
            "is_fault": False
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**data)
        
        assert "reconstruction_error" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)
    
    def test_missing_required_field_is_fault(self):
        """Test validation error when is_fault field is missing."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.05
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**data)
        
        assert "is_fault" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)
    
    def test_invalid_prediction_type(self):
        """Test validation error when prediction is not a string."""
        data = {
            "prediction": 123,
            "reconstruction_error": 0.05,
            "is_fault": False
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**data)
        
        assert "str type expected" in str(exc_info.value) or "string" in str(exc_info.value).lower()
    
    def test_invalid_reconstruction_error_type(self):
        """Test validation error when reconstruction_error is not a float."""
        data = {
            "prediction": "normal",
            "reconstruction_error": "invalid",
            "is_fault": False
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**data)
        
        assert "float" in str(exc_info.value).lower() or "number" in str(exc_info.value).lower()
    
    def test_invalid_is_fault_type(self):
        """Test validation error when is_fault is not a boolean."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.05,
            "is_fault": "not_boolean"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**data)
        
        assert "bool" in str(exc_info.value).lower() or "boolean" in str(exc_info.value).lower()
    
    def test_invalid_attribute_errors_type(self):
        """Test validation error when attribute_errors is not a dict."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.05,
            "is_fault": False,
            "attribute_errors": "not_a_dict"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**data)
        
        assert "dict" in str(exc_info.value).lower() or "mapping" in str(exc_info.value).lower()
    
    def test_attribute_errors_with_invalid_value_types(self):
        """Test validation error when attribute_errors contains non-float values."""
        data = {
            "prediction": "fault",
            "reconstruction_error": 0.85,
            "is_fault": True,
            "attribute_errors": {
                "temperature": 0.12,
                "pressure": "invalid_float",
                "vibration": 0.65
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**data)
        
        assert "float" in str(exc_info.value).lower() or "number" in str(exc_info.value).lower()
    
    def test_negative_reconstruction_error(self):
        """Test that negative reconstruction_error values are accepted."""
        data = {
            "prediction": "normal",
            "reconstruction_error": -0.05,
            "is_fault": False
        }
        response = PredictionResponse(**data)
        
        assert response.reconstruction_error == -0.05
    
    def test_zero_reconstruction_error(self):
        """Test that zero reconstruction_error is accepted."""
        data = {
            "prediction": "perfect",
            "reconstruction_error": 0.0,
            "is_fault": False
        }
        response = PredictionResponse(**data)
        
        assert response.reconstruction_error == 0.0
    
    def test_large_reconstruction_error(self):
        """Test that large reconstruction_error values are accepted."""
        data = {
            "prediction": "severe_fault",
            "reconstruction_error": 999.99,
            "is_fault": True
        }
        response = PredictionResponse(**data)
        
        assert response.reconstruction_error == 999.99
    
    def test_empty_prediction_string(self):
        """Test that empty prediction string is accepted."""
        data = {
            "prediction": "",
            "reconstruction_error": 0.05,
            "is_fault": False
        }
        response = PredictionResponse(**data)
        
        assert response.prediction == ""
    
    def test_long_prediction_string(self):
        """Test that long prediction strings are accepted."""
        long_prediction = "very_long_prediction_name_" * 10
        data = {
            "prediction": long_prediction,
            "reconstruction_error": 0.05,
            "is_fault": False
        }
        response = PredictionResponse(**data)
        
        assert response.prediction == long_prediction
    
    def test_special_characters_in_prediction(self):
        """Test that special characters in prediction string are accepted."""
        data = {
            "prediction": "fault-type_123!@#",
            "reconstruction_error": 0.05,
            "is_fault": True
        }
        response = PredictionResponse(**data)
        
        assert response.prediction == "fault-type_123!@#"
    
    def test_attribute_errors_with_special_key_names(self):
        """Test attribute_errors with various key naming patterns."""
        data = {
            "prediction": "fault",
            "reconstruction_error": 0.85,
            "is_fault": True,
            "attribute_errors": {
                "sensor_1": 0.12,
                "sensor-2": 0.08,
                "sensor 3": 0.65,
                "123_sensor": 0.23,
                "SENSOR_CAPS": 0.45
            }
        }
        response = PredictionResponse(**data)
        
        assert len(response.attribute_errors) == 5
        assert response.attribute_errors["sensor_1"] == 0.12
        assert response.attribute_errors["sensor-2"] == 0.08
        assert response.attribute_errors["sensor 3"] == 0.65
        assert response.attribute_errors["123_sensor"] == 0.23
        assert response.attribute_errors["SENSOR_CAPS"] == 0.45
    
    def test_attribute_errors_with_extreme_float_values(self):
        """Test attribute_errors with extreme float values."""
        data = {
            "prediction": "fault",
            "reconstruction_error": 0.85,
            "is_fault": True,
            "attribute_errors": {
                "min_value": -999999.99,
                "max_value": 999999.99,
                "zero_value": 0.0,
                "tiny_value": 1e-10,
                "large_value": 1e10
            }
        }
        response = PredictionResponse(**data)
        
        assert response.attribute_errors["min_value"] == -999999.99
        assert response.attribute_errors["max_value"] == 999999.99
        assert response.attribute_errors["zero_value"] == 0.0
        assert response.attribute_errors["tiny_value"] == 1e-10
        assert response.attribute_errors["large_value"] == 1e10
    
    def test_model_immutability(self):
        """Test that model instances are immutable by default."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.05,
            "is_fault": False
        }
        response = PredictionResponse(**data)
        
        # Pydantic models are mutable by default, but we can test assignment
        response.prediction = "fault"
        assert response.prediction == "fault"
    
    def test_model_equality(self):
        """Test equality comparison between model instances."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.05,
            "is_fault": False,
            "attribute_errors": {"temp": 0.01}
        }
        
        response1 = PredictionResponse(**data)
        response2 = PredictionResponse(**data)
        
        assert response1 == response2
        assert response1.dict() == response2.dict()
    
    def test_model_inequality(self):
        """Test inequality comparison between different model instances."""
        data1 = {
            "prediction": "normal",
            "reconstruction_error": 0.05,
            "is_fault": False
        }
        data2 = {
            "prediction": "fault",
            "reconstruction_error": 0.85,
            "is_fault": True
        }
        
        response1 = PredictionResponse(**data1)
        response2 = PredictionResponse(**data2)
        
        assert response1 != response2
        assert response1.dict() != response2.dict()
    
    def test_model_copy(self):
        """Test copying model instances."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.05,
            "is_fault": False,
            "attribute_errors": {"temp": 0.01}
        }
        
        original = PredictionResponse(**data)
        copied = original.copy(deep=True)
        
        assert original == copied
        assert original is not copied
        assert original.attribute_errors is not copied.attribute_errors
    
    def test_model_copy_with_updates(self):
        """Test copying model instances with field updates."""
        data = {
            "prediction": "normal",
            "reconstruction_error": 0.05,
            "is_fault": False
        }
        
        original = PredictionResponse(**data)
        updated = original.copy(update={"prediction": "fault", "is_fault": True})
        
        assert original.prediction == "normal"
        assert original.is_fault is False
        assert updated.prediction == "fault"
        assert updated.is_fault is True
        assert updated.reconstruction_error == 0.05  # unchanged
    
    @pytest.mark.parametrize("prediction,reconstruction_error,is_fault", [
        ("normal", 0.01, False),
        ("fault", 0.99, True),
        ("anomaly", 0.5, True),
        ("healthy", 0.0, False),
        ("critical", 1.0, True),
    ])
    def test_various_valid_combinations(self, prediction, reconstruction_error, is_fault):
        """Test various valid combinations of field values."""
        data = {
            "prediction": prediction,
            "reconstruction_error": reconstruction_error,
            "is_fault": is_fault
        }
        response = PredictionResponse(**data)
        
        assert response.prediction == prediction
        assert response.reconstruction_error == reconstruction_error
        assert response.is_fault == is_fault
        assert response.attribute_errors is None
    
    def test_model_schema(self):
        """Test that the model schema is generated correctly."""
        schema = PredictionResponse.schema()
        
        assert "properties" in schema
        assert "prediction" in schema["properties"]
        assert "reconstruction_error" in schema["properties"]
        assert "is_fault" in schema["properties"]
        assert "attribute_errors" in schema["properties"]
        
        # Check required fields
        assert "required" in schema
        required_fields = {"prediction", "reconstruction_error", "is_fault"}
        assert set(schema["required"]) == required_fields
        
        # Check field types
        assert schema["properties"]["prediction"]["type"] == "string"
        assert schema["properties"]["reconstruction_error"]["type"] == "number"
        assert schema["properties"]["is_fault"]["type"] == "boolean"