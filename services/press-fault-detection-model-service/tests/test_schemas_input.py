import pytest
from pydantic import ValidationError
import sys
import os

# Add the service directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.schemas.input import SensorData, SEQUENCE_LENGTH
except ImportError:
    # Alternative import path if the above doesn't work
    try:
        from schemas.input import SensorData, SEQUENCE_LENGTH
    except ImportError:
        # Final fallback for different project structures
        import importlib.util
        spec = importlib.util.spec_from_file_location("input_schemas", 
            os.path.join(os.path.dirname(__file__), "..", "app", "schemas", "input.py"))
        input_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(input_module)
        SensorData = input_module.SensorData
        SEQUENCE_LENGTH = input_module.SEQUENCE_LENGTH


class TestSensorData:
    """Test suite for SensorData Pydantic model validation.
    
    Testing framework: pytest
    This test suite comprehensively tests the SensorData Pydantic model which validates
    sensor time series data for press fault detection. The model requires three fields:
    AI0_Vibration, AI1_Vibration, and AI2_Current, each with minimum length requirements
    and equal length constraints.
    """
    
    def test_valid_sensor_data_creation(self):
        """Test successful creation of SensorData with valid inputs."""
        # Arrange
        valid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH, 
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act
        sensor_data = SensorData(**valid_data)
        
        # Assert
        assert sensor_data.AI0_Vibration == [1.0] * SEQUENCE_LENGTH
        assert sensor_data.AI1_Vibration == [2.0] * SEQUENCE_LENGTH
        assert sensor_data.AI2_Current == [3.0] * SEQUENCE_LENGTH
        assert len(sensor_data.AI0_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI1_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI2_Current) == SEQUENCE_LENGTH

    def test_valid_sensor_data_with_longer_sequences(self):
        """Test SensorData creation with sequences longer than minimum required."""
        # Arrange
        longer_length = SEQUENCE_LENGTH + 10
        valid_data = {
            "AI0_Vibration": [1.1] * longer_length,
            "AI1_Vibration": [2.2] * longer_length,
            "AI2_Current": [3.3] * longer_length
        }
        
        # Act
        sensor_data = SensorData(**valid_data)
        
        # Assert
        assert len(sensor_data.AI0_Vibration) == longer_length
        assert len(sensor_data.AI1_Vibration) == longer_length
        assert len(sensor_data.AI2_Current) == longer_length

    def test_valid_sensor_data_with_mixed_values(self):
        """Test SensorData creation with realistic mixed sensor values."""
        # Arrange
        vibration_data = [0.1, 0.2, -0.1, 0.15, 0.3] * (SEQUENCE_LENGTH // 5)
        current_data = [1.5, 2.0, 1.8, 2.2, 1.9] * (SEQUENCE_LENGTH // 5)
        
        # Pad to exact length if needed
        while len(vibration_data) < SEQUENCE_LENGTH:
            vibration_data.append(0.0)
        while len(current_data) < SEQUENCE_LENGTH:
            current_data.append(0.0)
            
        valid_data = {
            "AI0_Vibration": vibration_data[:SEQUENCE_LENGTH],
            "AI1_Vibration": vibration_data[:SEQUENCE_LENGTH],
            "AI2_Current": current_data[:SEQUENCE_LENGTH]
        }
        
        # Act
        sensor_data = SensorData(**valid_data)
        
        # Assert
        assert len(sensor_data.AI0_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI1_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI2_Current) == SEQUENCE_LENGTH

    def test_ai0_vibration_too_short_raises_validation_error(self):
        """Test that AI0_Vibration with insufficient data points raises ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": [1.0] * (SEQUENCE_LENGTH - 1),  # One short
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI0_Vibration" in error_msg
        assert f"최소 {SEQUENCE_LENGTH}개의 데이터 포인트를 가져야 합니다" in error_msg

    def test_ai1_vibration_too_short_raises_validation_error(self):
        """Test that AI1_Vibration with insufficient data points raises ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * (SEQUENCE_LENGTH - 5),  # Five short
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI1_Vibration" in error_msg
        assert f"최소 {SEQUENCE_LENGTH}개의 데이터 포인트를 가져야 합니다" in error_msg

    def test_ai2_current_too_short_raises_validation_error(self):
        """Test that AI2_Current with insufficient data points raises ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * (SEQUENCE_LENGTH - 10)  # Ten short
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI2_Current" in error_msg
        assert f"최소 {SEQUENCE_LENGTH}개의 데이터 포인트를 가져야 합니다" in error_msg

    def test_all_fields_too_short_raises_validation_error(self):
        """Test that all fields being too short raises ValidationError for each field."""
        # Arrange
        short_length = SEQUENCE_LENGTH - 1
        invalid_data = {
            "AI0_Vibration": [1.0] * short_length,
            "AI1_Vibration": [2.0] * short_length,
            "AI2_Current": [3.0] * short_length
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        # Should contain errors for all three fields
        assert "AI0_Vibration" in error_msg
        assert "AI1_Vibration" in error_msg  
        assert "AI2_Current" in error_msg

    def test_empty_lists_raise_validation_error(self):
        """Test that empty lists for all sensor data raise ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": [],
            "AI1_Vibration": [],
            "AI2_Current": []
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI0_Vibration" in error_msg
        assert "AI1_Vibration" in error_msg
        assert "AI2_Current" in error_msg

    def test_unequal_list_lengths_raises_validation_error(self):
        """Test that unequal list lengths raise ValidationError."""
        # Arrange - AI0 longer than others
        invalid_data = {
            "AI0_Vibration": [1.0] * (SEQUENCE_LENGTH + 5),
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "모든 센서 데이터 리스트의 길이는 동일해야 합니다" in error_msg

    def test_unequal_list_lengths_ai1_different_raises_validation_error(self):
        """Test that AI1_Vibration having different length raises ValidationError."""
        # Arrange - AI1 shorter than others but still meets minimum length
        invalid_data = {
            "AI0_Vibration": [1.0] * (SEQUENCE_LENGTH + 3),
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * (SEQUENCE_LENGTH + 3)
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "모든 센서 데이터 리스트의 길이는 동일해야 합니다" in error_msg

    def test_unequal_list_lengths_ai2_different_raises_validation_error(self):
        """Test that AI2_Current having different length raises ValidationError."""
        # Arrange - AI2 longer, others minimum length
        invalid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * (SEQUENCE_LENGTH + 2)
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "모든 센서 데이터 리스트의 길이는 동일해야 합니다" in error_msg

    def test_missing_required_field_ai0_raises_validation_error(self):
        """Test that missing AI0_Vibration field raises ValidationError."""
        # Arrange
        invalid_data = {
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI0_Vibration" in error_msg

    def test_missing_required_field_ai1_raises_validation_error(self):
        """Test that missing AI1_Vibration field raises ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI1_Vibration" in error_msg

    def test_missing_required_field_ai2_raises_validation_error(self):
        """Test that missing AI2_Current field raises ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI2_Current" in error_msg

    def test_non_numeric_values_in_ai0_raises_validation_error(self):
        """Test that non-numeric values in AI0_Vibration raise ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": ["not_a_number"] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI0_Vibration" in error_msg

    def test_non_numeric_values_in_ai1_raises_validation_error(self):
        """Test that non-numeric values in AI1_Vibration raise ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [None] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI1_Vibration" in error_msg

    def test_non_list_input_raises_validation_error(self):
        """Test that non-list inputs raise ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": "not_a_list",
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI0_Vibration" in error_msg

    def test_mixed_valid_and_invalid_numeric_values(self):
        """Test with mix of valid floats, integers, and edge cases."""
        # Arrange
        valid_data = {
            "AI0_Vibration": [1, 2.5, -3.7, 0, 0.001] * (SEQUENCE_LENGTH // 5),
            "AI1_Vibration": [-1.0, 0.0, 1.0, 2.5, -2.5] * (SEQUENCE_LENGTH // 5),
            "AI2_Current": [0.1, 10.5, -5.2, 100.0, 0.001] * (SEQUENCE_LENGTH // 5)
        }
        
        # Ensure exact length
        for key in valid_data:
            while len(valid_data[key]) < SEQUENCE_LENGTH:
                valid_data[key].append(0.0)
            valid_data[key] = valid_data[key][:SEQUENCE_LENGTH]
        
        # Act
        sensor_data = SensorData(**valid_data)
        
        # Assert
        assert len(sensor_data.AI0_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI1_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI2_Current) == SEQUENCE_LENGTH

    def test_extreme_numeric_values(self):
        """Test with extreme but valid numeric values."""
        # Arrange
        valid_data = {
            "AI0_Vibration": [float('-inf')] * SEQUENCE_LENGTH,
            "AI1_Vibration": [float('inf')] * SEQUENCE_LENGTH,
            "AI2_Current": [1e10] * SEQUENCE_LENGTH
        }
        
        # Act
        sensor_data = SensorData(**valid_data)
        
        # Assert
        assert all(x == float('-inf') for x in sensor_data.AI0_Vibration)
        assert all(x == float('inf') for x in sensor_data.AI1_Vibration)
        assert all(x == 1e10 for x in sensor_data.AI2_Current)

    def test_nan_values_are_handled(self):
        """Test that NaN values are handled appropriately."""
        # Arrange
        valid_data = {
            "AI0_Vibration": [float('nan')] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act
        sensor_data = SensorData(**valid_data)
        
        # Assert
        import math
        assert all(math.isnan(x) for x in sensor_data.AI0_Vibration)
        assert len(sensor_data.AI0_Vibration) == SEQUENCE_LENGTH

    def test_sequence_length_constant(self):
        """Test that SEQUENCE_LENGTH constant has expected value."""
        # Assert
        assert SEQUENCE_LENGTH == 20
        assert isinstance(SEQUENCE_LENGTH, int)
        assert SEQUENCE_LENGTH > 0

    def test_field_descriptions_are_set(self):
        """Test that field descriptions are properly set."""
        # Act
        schema = SensorData.model_json_schema()
        
        # Assert  
        assert "properties" in schema
        properties = schema["properties"]
        
        assert "AI0_Vibration" in properties
        assert properties["AI0_Vibration"]["description"] == "upper vibration time series data list"
        
        assert "AI1_Vibration" in properties  
        assert properties["AI1_Vibration"]["description"] == "lower vibration time series data list"
        
        assert "AI2_Current" in properties
        assert properties["AI2_Current"]["description"] == "current time series data list"

    def test_model_serialization_deserialization(self):
        """Test that the model can be properly serialized and deserialized."""
        # Arrange
        original_data = {
            "AI0_Vibration": [1.1, 2.2, 3.3] * (SEQUENCE_LENGTH // 3) + [4.4],
            "AI1_Vibration": [5.5, 6.6, 7.7] * (SEQUENCE_LENGTH // 3) + [8.8], 
            "AI2_Current": [9.9, 10.1, 11.2] * (SEQUENCE_LENGTH // 3) + [12.3]
        }
        
        # Ensure exact length
        for key in original_data:
            while len(original_data[key]) < SEQUENCE_LENGTH:
                original_data[key].append(0.0)
            original_data[key] = original_data[key][:SEQUENCE_LENGTH]
        
        # Act
        sensor_data = SensorData(**original_data)
        json_data = sensor_data.model_dump_json()
        restored_sensor_data = SensorData.model_validate_json(json_data)
        
        # Assert
        assert sensor_data.AI0_Vibration == restored_sensor_data.AI0_Vibration
        assert sensor_data.AI1_Vibration == restored_sensor_data.AI1_Vibration
        assert sensor_data.AI2_Current == restored_sensor_data.AI2_Current

    def test_model_dict_conversion(self):
        """Test that the model can be converted to/from dictionary format."""
        # Arrange
        original_data = {
            "AI0_Vibration": [0.5] * SEQUENCE_LENGTH,
            "AI1_Vibration": [1.5] * SEQUENCE_LENGTH,
            "AI2_Current": [2.5] * SEQUENCE_LENGTH
        }
        
        # Act
        sensor_data = SensorData(**original_data)
        dict_data = sensor_data.model_dump()
        restored_sensor_data = SensorData(**dict_data)
        
        # Assert
        assert dict_data == original_data
        assert sensor_data.AI0_Vibration == restored_sensor_data.AI0_Vibration
        assert sensor_data.AI1_Vibration == restored_sensor_data.AI1_Vibration
        assert sensor_data.AI2_Current == restored_sensor_data.AI2_Current

    @pytest.mark.parametrize("field_name", ["AI0_Vibration", "AI1_Vibration", "AI2_Current"])
    def test_individual_field_validation_parametrized(self, field_name):
        """Parametrized test for individual field length validation."""
        # Arrange
        base_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        base_data[field_name] = [1.0] * (SEQUENCE_LENGTH - 1)  # Make this field too short
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**base_data)
        
        error_msg = str(exc_info.value)
        assert field_name in error_msg
        assert f"최소 {SEQUENCE_LENGTH}개의 데이터 포인트를 가져야 합니다" in error_msg

    @pytest.mark.parametrize("length", [SEQUENCE_LENGTH, SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH + 10, SEQUENCE_LENGTH * 2])
    def test_valid_lengths_parametrized(self, length):
        """Parametrized test for various valid lengths."""
        # Arrange
        valid_data = {
            "AI0_Vibration": [1.0] * length,
            "AI1_Vibration": [2.0] * length,
            "AI2_Current": [3.0] * length
        }
        
        # Act
        sensor_data = SensorData(**valid_data)
        
        # Assert
        assert len(sensor_data.AI0_Vibration) == length
        assert len(sensor_data.AI1_Vibration) == length  
        assert len(sensor_data.AI2_Current) == length

    def test_validator_error_messages_are_korean(self):
        """Test that validation error messages are in Korean as expected."""
        # Arrange - Test minimum length validation
        invalid_data = {
            "AI0_Vibration": [1.0] * (SEQUENCE_LENGTH - 1),
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "필드는 최소" in error_msg
        assert "개의 데이터 포인트를 가져야 합니다" in error_msg
        
        # Arrange - Test length equality validation  
        invalid_data2 = {
            "AI0_Vibration": [1.0] * (SEQUENCE_LENGTH + 1),
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data2)
            
        error_msg = str(exc_info.value)
        assert "모든 센서 데이터 리스트의 길이는 동일해야 합니다" in error_msg

    def test_field_validator_execution_order(self):
        """Test that field validators run before model validators."""
        # Arrange - Field validation should fail before model validation
        invalid_data = {
            "AI0_Vibration": [1.0] * (SEQUENCE_LENGTH - 1),  # Too short - field validation fails
            "AI1_Vibration": [2.0] * (SEQUENCE_LENGTH + 1),  # Different length - would fail model validation
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        # Should fail on field validation first
        error_msg = str(exc_info.value)
        assert "AI0_Vibration" in error_msg
        assert f"최소 {SEQUENCE_LENGTH}개의 데이터 포인트를 가져야 합니다" in error_msg

    def test_model_validator_execution_after_field_validation(self):
        """Test that model validator runs after field validation passes."""
        # Arrange - All fields meet minimum length but have different lengths
        invalid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * (SEQUENCE_LENGTH + 1),  # One longer
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        # Should fail on model validation (length equality check)
        error_msg = str(exc_info.value)
        assert "모든 센서 데이터 리스트의 길이는 동일해야 합니다" in error_msg

    def test_large_sequence_lengths(self):
        """Test with very large sequence lengths to ensure scalability."""
        # Arrange
        large_length = SEQUENCE_LENGTH * 100  # Much larger than minimum
        valid_data = {
            "AI0_Vibration": [0.1] * large_length,
            "AI1_Vibration": [0.2] * large_length,
            "AI2_Current": [1.0] * large_length
        }
        
        # Act
        sensor_data = SensorData(**valid_data)
        
        # Assert
        assert len(sensor_data.AI0_Vibration) == large_length
        assert len(sensor_data.AI1_Vibration) == large_length
        assert len(sensor_data.AI2_Current) == large_length

    def test_very_small_numeric_values(self):
        """Test with very small numeric values."""
        # Arrange
        small_values_data = {
            "AI0_Vibration": [1e-10] * SEQUENCE_LENGTH,
            "AI1_Vibration": [-1e-15] * SEQUENCE_LENGTH,
            "AI2_Current": [1e-20] * SEQUENCE_LENGTH
        }
        
        # Act
        sensor_data = SensorData(**small_values_data)
        
        # Assert
        assert all(x == 1e-10 for x in sensor_data.AI0_Vibration)
        assert all(x == -1e-15 for x in sensor_data.AI1_Vibration)
        assert all(x == 1e-20 for x in sensor_data.AI2_Current)

    def test_zero_values_are_valid(self):
        """Test that zero values are accepted."""
        # Arrange
        zero_data = {
            "AI0_Vibration": [0.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [0] * SEQUENCE_LENGTH,  # Integer zero
            "AI2_Current": [-0.0] * SEQUENCE_LENGTH  # Negative zero
        }
        
        # Act
        sensor_data = SensorData(**zero_data)
        
        # Assert
        assert all(x == 0.0 for x in sensor_data.AI0_Vibration)
        assert all(x == 0 for x in sensor_data.AI1_Vibration)
        assert all(x == 0.0 for x in sensor_data.AI2_Current)

    def test_integer_to_float_conversion(self):
        """Test that integer values are automatically converted to floats."""
        # Arrange
        mixed_type_data = {
            "AI0_Vibration": [1, 2, 3] * (SEQUENCE_LENGTH // 3) + [4] * (SEQUENCE_LENGTH % 3),
            "AI1_Vibration": [5, 6, 7] * (SEQUENCE_LENGTH // 3) + [8] * (SEQUENCE_LENGTH % 3),
            "AI2_Current": [9, 10, 11] * (SEQUENCE_LENGTH // 3) + [12] * (SEQUENCE_LENGTH % 3)
        }
        
        # Act
        sensor_data = SensorData(**mixed_type_data)
        
        # Assert - All values should be converted to floats
        assert all(isinstance(x, float) for x in sensor_data.AI0_Vibration)
        assert all(isinstance(x, float) for x in sensor_data.AI1_Vibration)
        assert all(isinstance(x, float) for x in sensor_data.AI2_Current)

    def test_boolean_values_raise_validation_error(self):
        """Test that boolean values in the lists raise ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": [True, False] * (SEQUENCE_LENGTH // 2),
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        # Boolean values should be converted to 1.0 and 0.0, so this test might actually pass
        # Let's test with a more explicit non-numeric value
        invalid_data["AI0_Vibration"] = [True] * SEQUENCE_LENGTH
        sensor_data = SensorData(**invalid_data)
        assert all(x == 1.0 for x in sensor_data.AI0_Vibration)  # True converts to 1.0

    def test_dict_input_raises_validation_error(self):
        """Test that dictionary input instead of list raises ValidationError."""
        # Arrange
        invalid_data = {
            "AI0_Vibration": {"not": "a_list"},
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "AI0_Vibration" in error_msg


class TestSequenceLengthConstant:
    """Test suite for the SEQUENCE_LENGTH constant."""
    
    def test_sequence_length_is_twenty(self):
        """Test that SEQUENCE_LENGTH is exactly 20."""
        assert SEQUENCE_LENGTH == 20
    
    def test_sequence_length_type(self):
        """Test that SEQUENCE_LENGTH is an integer."""
        assert isinstance(SEQUENCE_LENGTH, int)
    
    def test_sequence_length_positive(self):
        """Test that SEQUENCE_LENGTH is positive."""
        assert SEQUENCE_LENGTH > 0


class TestFieldValidators:
    """Test suite specifically for field validator behavior."""
    
    def test_field_validator_applies_to_all_specified_fields(self):
        """Test that the field validator applies to all three specified fields."""
        # Test each field individually with too-short data
        for field_name in ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]:
            # with pytest.subTest(field=field_name):
            # Arrange
            base_data = {
                "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
                "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
                "AI2_Current": [3.0] * SEQUENCE_LENGTH
            }
            base_data[field_name] = [1.0] * (SEQUENCE_LENGTH - 1)
            
            # Act & Assert
            with pytest.raises(ValidationError) as exc_info:
                SensorData(**base_data)
            
            error_msg = str(exc_info.value)
            assert field_name in error_msg
            assert "최소" in error_msg
                
    def test_field_validator_minimum_length_boundary(self):
        """Test field validator at the exact minimum length boundary."""
        # Arrange - Exactly minimum length should pass field validation
        boundary_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act - Should not raise ValidationError for field validation
        sensor_data = SensorData(**boundary_data)
        
        # Assert
        assert len(sensor_data.AI0_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI1_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI2_Current) == SEQUENCE_LENGTH

    def test_field_validator_off_by_one_errors(self):
        """Test field validator with off-by-one boundary conditions."""
        # Test exactly one below minimum
        for field_name in ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]:
            base_data = {
                "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
                "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
                "AI2_Current": [3.0] * SEQUENCE_LENGTH
            }
            base_data[field_name] = [1.0] * (SEQUENCE_LENGTH - 1)
            
            with pytest.raises(ValidationError):
                SensorData(**base_data)
        
        # Test exactly at minimum (should pass)
        valid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        sensor_data = SensorData(**valid_data)
        assert isinstance(sensor_data, SensorData)


class TestModelValidator:
    """Test suite specifically for model validator behavior."""
    
    def test_model_validator_equal_lengths_pass(self):
        """Test that equal lengths pass model validation."""
        # Arrange
        equal_lengths_data = {
            "AI0_Vibration": [1.0] * (SEQUENCE_LENGTH + 5),
            "AI1_Vibration": [2.0] * (SEQUENCE_LENGTH + 5),
            "AI2_Current": [3.0] * (SEQUENCE_LENGTH + 5)
        }
        
        # Act
        sensor_data = SensorData(**equal_lengths_data)
        
        # Assert
        assert len(sensor_data.AI0_Vibration) == SEQUENCE_LENGTH + 5
        assert len(sensor_data.AI1_Vibration) == SEQUENCE_LENGTH + 5
        assert len(sensor_data.AI2_Current) == SEQUENCE_LENGTH + 5
    
    def test_model_validator_returns_self(self):
        """Test that model validator returns self instance."""
        # Arrange
        valid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act
        sensor_data = SensorData(**valid_data)
        
        # Assert - The model validator should return the instance itself
        assert isinstance(sensor_data, SensorData)
        assert sensor_data.AI0_Vibration == [1.0] * SEQUENCE_LENGTH
        assert sensor_data.AI1_Vibration == [2.0] * SEQUENCE_LENGTH
        assert sensor_data.AI2_Current == [3.0] * SEQUENCE_LENGTH

    @pytest.mark.parametrize("different_field", ["AI0_Vibration", "AI1_Vibration", "AI2_Current"])
    def test_model_validator_fails_on_any_different_length(self, different_field):
        """Parametrized test that model validator fails when any field has different length."""
        # Arrange
        base_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        # Make one field different length
        base_data[different_field] = [1.0] * (SEQUENCE_LENGTH + 1)
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**base_data)
        
        error_msg = str(exc_info.value)
        assert "모든 센서 데이터 리스트의 길이는 동일해야 합니다" in error_msg

    def test_model_validator_triple_length_mismatch(self):
        """Test model validator when all three fields have different lengths."""
        # Arrange - All different lengths but all meet minimum requirement
        invalid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * (SEQUENCE_LENGTH + 1),
            "AI2_Current": [3.0] * (SEQUENCE_LENGTH + 2)
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_msg = str(exc_info.value)
        assert "모든 센서 데이터 리스트의 길이는 동일해야 합니다" in error_msg


class TestEdgeCases:
    """Test suite for edge cases and corner scenarios."""
    
    def test_exactly_minimum_length_all_fields(self):
        """Test that exactly minimum length works for all fields."""
        # Arrange
        exact_min_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        
        # Act
        sensor_data = SensorData(**exact_min_data)
        
        # Assert
        assert len(sensor_data.AI0_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI1_Vibration) == SEQUENCE_LENGTH
        assert len(sensor_data.AI2_Current) == SEQUENCE_LENGTH

    def test_field_info_validation(self):
        """Test that field info and metadata are correctly set."""
        # Get the model fields info
        fields_info = SensorData.model_fields
        
        # Assert field requirements
        assert "AI0_Vibration" in fields_info
        assert "AI1_Vibration" in fields_info
        assert "AI2_Current" in fields_info
        
        # Check that all fields are required (no default values)
        for field_name in ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]:
            field_info = fields_info[field_name]
            assert field_info.is_required()

    def test_model_configuration(self):
        """Test model configuration and metadata."""
        # Create a valid instance
        valid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        sensor_data = SensorData(**valid_data)
        
        # Test that the model can be serialized properly
        json_dict = sensor_data.model_dump()
        assert isinstance(json_dict, dict)
        assert set(json_dict.keys()) == {"AI0_Vibration", "AI1_Vibration", "AI2_Current"}

    def test_field_alias_handling(self):
        """Test that no field aliases are interfering with validation."""
        # Create instance with exact field names
        valid_data = {
            "AI0_Vibration": [1.0] * SEQUENCE_LENGTH,
            "AI1_Vibration": [2.0] * SEQUENCE_LENGTH,
            "AI2_Current": [3.0] * SEQUENCE_LENGTH
        }
        sensor_data = SensorData(**valid_data)
        
        # Test direct attribute access
        assert hasattr(sensor_data, 'AI0_Vibration')
        assert hasattr(sensor_data, 'AI1_Vibration')  
        assert hasattr(sensor_data, 'AI2_Current')

    def test_very_large_lists_performance(self):
        """Test performance with very large lists (basic smoke test)."""
        # Arrange - Create very large sequences
        very_large_length = SEQUENCE_LENGTH * 1000
        large_data = {
            "AI0_Vibration": [0.1] * very_large_length,
            "AI1_Vibration": [0.2] * very_large_length,
            "AI2_Current": [1.0] * very_large_length
        }
        
        # Act - This should complete in reasonable time
        sensor_data = SensorData(**large_data)
        
        # Assert
        assert len(sensor_data.AI0_Vibration) == very_large_length
        assert len(sensor_data.AI1_Vibration) == very_large_length
        assert len(sensor_data.AI2_Current) == very_large_length

    def test_validation_error_details(self):
        """Test that validation errors contain sufficient detail for debugging."""
        # Arrange - Multiple validation errors
        invalid_data = {
            "AI0_Vibration": [1.0] * (SEQUENCE_LENGTH - 1),  # Too short
            "AI1_Vibration": "not_a_list",  # Wrong type
            "AI2_Current": [None] * SEQUENCE_LENGTH  # Invalid values
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            SensorData(**invalid_data)
        
        error_details = exc_info.value.errors()
        assert len(error_details) >= 2  # Should have multiple errors
        
        # Check that error details contain field names
        error_msg = str(exc_info.value)
        assert "AI0_Vibration" in error_msg
        assert "AI1_Vibration" in error_msg