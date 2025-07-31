import pytest
from pydantic import ValidationError

# Import the schema to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schemas.input import SensorData


class TestSensorData:
    """Comprehensive unit tests for SensorData schema validation."""
    
    def test_valid_current_sensor_data(self):
        """Test valid current sensor data creation."""
        data = {
            "signal_type": "cur",
            "values": [1.0, 2.5, 3.7, 4.2]
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "cur"
        assert sensor.values == [1.0, 2.5, 3.7, 4.2]
    
    def test_valid_vibration_sensor_data(self):
        """Test valid vibration sensor data creation."""
        data = {
            "signal_type": "vib",
            "values": [0.1, 0.2, 0.15, 0.25]
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "vib"
        assert sensor.values == [0.1, 0.2, 0.15, 0.25]
    
    def test_empty_values_list(self):
        """Test sensor data with empty values list."""
        data = {
            "signal_type": "cur",
            "values": []
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "cur"
        assert sensor.values == []
    
    def test_single_value(self):
        """Test sensor data with single value."""
        data = {
            "signal_type": "vib",
            "values": [42.0]
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "vib"
        assert sensor.values == [42.0]
    
    def test_large_values_list(self):
        """Test sensor data with large number of values."""
        values = [float(i) for i in range(1000)]
        data = {
            "signal_type": "cur",
            "values": values
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "cur"
        assert len(sensor.values) == 1000
        assert sensor.values == values
    
    def test_negative_values(self):
        """Test sensor data with negative values."""
        data = {
            "signal_type": "cur",
            "values": [-1.0, -2.5, -3.7]
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "cur"
        assert sensor.values == [-1.0, -2.5, -3.7]
    
    def test_zero_values(self):
        """Test sensor data with zero values."""
        data = {
            "signal_type": "vib",
            "values": [0.0, 0.0, 0.0]
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "vib"
        assert sensor.values == [0.0, 0.0, 0.0]
    
    def test_very_large_values(self):
        """Test sensor data with very large float values."""
        data = {
            "signal_type": "cur",
            "values": [1e10, 1e-10, float('inf')]
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "cur"
        assert sensor.values[0] == 1e10
        assert sensor.values[1] == 1e-10
        assert sensor.values[2] == float('inf')
    
    def test_integer_values_converted_to_float(self):
        """Test that integer values are accepted and converted to float."""
        data = {
            "signal_type": "cur",
            "values": [1, 2, 3, 4]
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "cur"
        assert sensor.values == [1.0, 2.0, 3.0, 4.0]
        assert all(isinstance(v, float) for v in sensor.values)
    
    def test_mixed_int_float_values(self):
        """Test sensor data with mixed integer and float values."""
        data = {
            "signal_type": "vib",
            "values": [1, 2.5, 3, 4.7]
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "vib"
        assert sensor.values == [1.0, 2.5, 3.0, 4.7]
    
    def test_invalid_signal_type(self):
        """Test that invalid signal types raise ValidationError."""
        data = {
            "signal_type": "invalid",
            "values": [1.0, 2.0]
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "literal_error"
        assert "signal_type" in errors[0]["loc"]
    
    def test_missing_signal_type(self):
        """Test that missing signal_type raises ValidationError."""
        data = {
            "values": [1.0, 2.0]
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert "signal_type" in errors[0]["loc"]
    
    def test_missing_values(self):
        """Test that missing values raises ValidationError."""
        data = {
            "signal_type": "cur"
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "missing"
        assert "values" in errors[0]["loc"]
    
    def test_none_signal_type(self):
        """Test that None signal_type raises ValidationError."""
        data = {
            "signal_type": None,
            "values": [1.0, 2.0]
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "signal_type" in errors[0]["loc"]
    
    def test_none_values(self):
        """Test that None values raises ValidationError."""
        data = {
            "signal_type": "cur",
            "values": None
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "values" in errors[0]["loc"]
    
    def test_string_values_invalid(self):
        """Test that string values in the list raise ValidationError."""
        data = {
            "signal_type": "cur",
            "values": ["1.0", "2.0", "3.0"]
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any("values" in error["loc"] for error in errors)
    
    def test_mixed_valid_invalid_values(self):
        """Test that mixed valid and invalid values raise ValidationError."""
        data = {
            "signal_type": "vib",
            "values": [1.0, "invalid", 3.0]
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any("values" in error["loc"] for error in errors)
    
    def test_non_list_values(self):
        """Test that non-list values raise ValidationError."""
        data = {
            "signal_type": "cur",
            "values": "not a list"
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "values" in errors[0]["loc"]
    
    def test_dict_values_invalid(self):
        """Test that dictionary values raise ValidationError."""
        data = {
            "signal_type": "cur",
            "values": {"key": "value"}
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "values" in errors[0]["loc"]
    
    def test_case_sensitive_signal_type(self):
        """Test that signal_type is case sensitive."""
        data = {
            "signal_type": "CUR",
            "values": [1.0, 2.0]
        }
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**data)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "literal_error"
    
    def test_extra_fields_ignored(self):
        """Test that extra fields are ignored by default."""
        data = {
            "signal_type": "cur",
            "values": [1.0, 2.0],
            "extra_field": "should be ignored"
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == "cur"
        assert sensor.values == [1.0, 2.0]
        assert not hasattr(sensor, "extra_field")
    
    def test_model_serialization(self):
        """Test that the model can be serialized to dict."""
        data = {
            "signal_type": "vib",
            "values": [1.0, 2.0, 3.0]
        }
        sensor = SensorData(**data)
        serialized = sensor.model_dump()
        
        assert serialized == data
        assert isinstance(serialized, dict)
    
    def test_model_json_serialization(self):
        """Test that the model can be serialized to JSON."""
        data = {
            "signal_type": "cur",
            "values": [1.5, 2.5, 3.5]
        }
        sensor = SensorData(**data)
        json_str = sensor.model_dump_json()
        
        assert isinstance(json_str, str)
        assert "cur" in json_str
        assert "1.5" in json_str
    
    def test_model_from_json(self):
        """Test that the model can be created from JSON."""
        json_str = '{"signal_type": "vib", "values": [1.0, 2.0, 3.0]}'
        sensor = SensorData.model_validate_json(json_str)
        
        assert sensor.signal_type == "vib"
        assert sensor.values == [1.0, 2.0, 3.0]
    
    def test_model_equality(self):
        """Test model equality comparison."""
        data = {
            "signal_type": "cur",
            "values": [1.0, 2.0]
        }
        sensor1 = SensorData(**data)
        sensor2 = SensorData(**data)
        
        assert sensor1 == sensor2
    
    def test_model_inequality(self):
        """Test model inequality comparison."""
        sensor1 = SensorData(signal_type="cur", values=[1.0, 2.0])
        sensor2 = SensorData(signal_type="vib", values=[1.0, 2.0])
        sensor3 = SensorData(signal_type="cur", values=[3.0, 4.0])
        
        assert sensor1 != sensor2
        assert sensor1 != sensor3
    
    def test_model_hash(self):
        """Test that model instances are hashable."""
        sensor = SensorData(signal_type="cur", values=[1.0, 2.0])
        hash_value = hash(sensor)
        
        assert isinstance(hash_value, int)
    
    def test_model_repr(self):
        """Test model string representation."""
        sensor = SensorData(signal_type="cur", values=[1.0, 2.0])
        repr_str = repr(sensor)
        
        assert "SensorData" in repr_str
        assert "cur" in repr_str
    
    def test_field_access(self):
        """Test direct field access."""
        sensor = SensorData(signal_type="vib", values=[1.0, 2.0, 3.0])
        
        assert sensor.signal_type == "vib"
        assert sensor.values == [1.0, 2.0, 3.0]
        assert len(sensor.values) == 3
    
    def test_immutability_attempt(self):
        """Test that attempting to modify fields after creation fails or is ignored."""
        sensor = SensorData(signal_type="cur", values=[1.0, 2.0])
        
        # Pydantic models are mutable by default, but we can test current behavior
        
        # These should work since Pydantic models are mutable by default
        sensor.signal_type = "vib"
        sensor.values.append(3.0)
        
        # Verify the changes took effect (documenting current behavior)
        assert sensor.signal_type == "vib"
        assert len(sensor.values) == 3
    
    @pytest.mark.parametrize("signal_type", ["cur", "vib"])
    def test_both_signal_types_parametrized(self, signal_type):
        """Parametrized test for both valid signal types."""
        data = {
            "signal_type": signal_type,
            "values": [1.0, 2.0, 3.0]
        }
        sensor = SensorData(**data)
        assert sensor.signal_type == signal_type
        assert sensor.values == [1.0, 2.0, 3.0]
    
    @pytest.mark.parametrize("invalid_type", ["current", "vibration", "CURRENT", "VIBRATION", "c", "v", "", " ", "cur ", " cur"])
    def test_invalid_signal_types_parametrized(self, invalid_type):
        """Parametrized test for various invalid signal types."""
        data = {
            "signal_type": invalid_type,
            "values": [1.0, 2.0]
        }
        with pytest.raises(ValidationError):
            SensorData(**data)
    
    @pytest.mark.parametrize("values", [
        [1.0],
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [-1.0, 0.0, 1.0],
        [1e-10, 1e10],
        []
    ])
    def test_various_value_lists_parametrized(self, values):
        """Parametrized test for various valid value lists."""
        data = {
            "signal_type": "cur",
            "values": values
        }
        sensor = SensorData(**data)
        assert sensor.values == values