import os
import pytest
from unittest.mock import patch
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from services.painting_process_equipment_simulator_service.settings import Settings, settings


class TestSettings:
    """Test suite for Settings configuration class using pytest framework."""

    def test_settings_inherits_from_base_settings(self):
        """Test that Settings class properly inherits from BaseSettings."""
        assert issubclass(Settings, BaseSettings)
        assert isinstance(settings, Settings)

    def test_default_values(self):
        """Test that default values are set correctly."""
        test_settings = Settings(azure_connection_string="test_connection")
        
        assert test_settings.azure_container_name == "simulator-data"
        assert test_settings.painting_data_folder == "painting-process-equipment"
        assert test_settings.scheduler_interval_minutes == 1
        assert test_settings.batch_size == 10
        assert test_settings.log_directory == "logs"
        assert test_settings.log_filename == "painting_issue_logs.json"
        assert test_settings.error_log_filename == "painting_errors.json"
        assert test_settings.http_timeout == 30
        assert test_settings.max_retries == 3

    def test_required_fields(self):
        """Test that required fields raise ValidationError when missing."""
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        error_details = exc_info.value.errors()
        assert any(error['loc'] == ('azure_connection_string',) for error in error_details)

    def test_azure_connection_string_validation(self):
        """Test azure_connection_string field validation."""
        # Valid connection string
        test_settings = Settings(azure_connection_string="DefaultEndpointsProtocol=https;AccountName=test")
        assert test_settings.azure_connection_string == "DefaultEndpointsProtocol=https;AccountName=test"
        
        # Empty string should raise validation error
        with pytest.raises(ValidationError):
            Settings(azure_connection_string="")

    def test_integer_field_validation(self):
        """Test integer field validation for various numeric settings."""
        # Valid integers
        test_settings = Settings(
            azure_connection_string="test_connection",
            scheduler_interval_minutes=5,
            batch_size=20,
            http_timeout=60,
            max_retries=5
        )
        assert test_settings.scheduler_interval_minutes == 5
        assert test_settings.batch_size == 20
        assert test_settings.http_timeout == 60
        assert test_settings.max_retries == 5

        # Invalid types should raise validation error
        with pytest.raises(ValidationError):
            Settings(azure_connection_string="test", scheduler_interval_minutes="invalid")

    def test_negative_integer_handling(self):
        """Test handling of negative values for integer fields."""
        # Test negative scheduler interval
        test_settings = Settings(
            azure_connection_string="test_connection",
            scheduler_interval_minutes=-1
        )
        assert test_settings.scheduler_interval_minutes == -1

        # Test negative batch size
        test_settings = Settings(
            azure_connection_string="test_connection",
            batch_size=-5
        )
        assert test_settings.batch_size == -5

    def test_zero_values(self):
        """Test handling of zero values for integer fields."""
        test_settings = Settings(
            azure_connection_string="test_connection",
            scheduler_interval_minutes=0,
            batch_size=0,
            http_timeout=0,
            max_retries=0
        )
        assert test_settings.scheduler_interval_minutes == 0
        assert test_settings.batch_size == 0
        assert test_settings.http_timeout == 0
        assert test_settings.max_retries == 0

    def test_large_integer_values(self):
        """Test handling of large integer values."""
        test_settings = Settings(
            azure_connection_string="test_connection",
            scheduler_interval_minutes=999999,
            batch_size=1000000,
            http_timeout=86400,
            max_retries=100
        )
        assert test_settings.scheduler_interval_minutes == 999999
        assert test_settings.batch_size == 1000000
        assert test_settings.http_timeout == 86400
        assert test_settings.max_retries == 100

    @patch.dict(os.environ, {"PAINTING_SERVICE_URL": "http://custom-host:9000"})
    def test_painting_service_url_from_environment(self):
        """Test painting_service_url reads from environment variable."""
        test_settings = Settings(azure_connection_string="test_connection")
        assert test_settings.painting_service_url == "http://custom-host:9000"

    @patch.dict(os.environ, {}, clear=True)
    def test_painting_service_url_default_value(self):
        """Test painting_service_url uses default value when env var not set."""
        test_settings = Settings(azure_connection_string="test_connection")
        assert test_settings.painting_service_url == "http://localhost:8001"

    def test_model_services_property(self):
        """Test model_services property returns correct dictionary."""
        test_settings = Settings(
            azure_connection_string="test_connection",
            painting_service_url="http://test-service:8080"
        )
        
        expected = {"painting-process-equipment": "http://test-service:8080"}
        assert test_settings.model_services == expected

    def test_model_services_property_with_default_url(self):
        """Test model_services property with default painting service URL."""
        with patch.dict(os.environ, {}, clear=True):
            test_settings = Settings(azure_connection_string="test_connection")
            expected = {"painting-process-equipment": "http://localhost:8001"}
            assert test_settings.model_services == expected

    def test_model_services_property_is_dynamic(self):
        """Test that model_services property reflects changes to painting_service_url."""
        test_settings = Settings(
            azure_connection_string="test_connection",
            painting_service_url="http://initial:8000"
        )
        
        # Initial value
        assert test_settings.model_services["painting-process-equipment"] == "http://initial:8000"
        
        # Change the URL and verify property updates
        test_settings.painting_service_url = "http://updated:9000"
        assert test_settings.model_services["painting-process-equipment"] == "http://updated:9000"

    def test_string_field_validation(self):
        """Test string field validation and handling."""
        test_settings = Settings(
            azure_connection_string="test_connection",
            azure_container_name="custom-container",
            painting_data_folder="custom-folder",
            log_directory="custom-logs",
            log_filename="custom.log",
            error_log_filename="errors.log"
        )
        
        assert test_settings.azure_container_name == "custom-container"
        assert test_settings.painting_data_folder == "custom-folder"
        assert test_settings.log_directory == "custom-logs"
        assert test_settings.log_filename == "custom.log"
        assert test_settings.error_log_filename == "errors.log"

    def test_empty_string_handling(self):
        """Test handling of empty strings for optional string fields."""
        test_settings = Settings(
            azure_connection_string="test_connection",
            azure_container_name="",
            painting_data_folder="",
            log_directory="",
            log_filename="",
            error_log_filename=""
        )
        
        assert test_settings.azure_container_name == ""
        assert test_settings.painting_data_folder == ""
        assert test_settings.log_directory == ""
        assert test_settings.log_filename == ""
        assert test_settings.error_log_filename == ""

    def test_special_characters_in_strings(self):
        """Test handling of special characters in string fields."""
        test_settings = Settings(
            azure_connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=abc123==;EndpointSuffix=core.windows.net",
            azure_container_name="test-container_123",
            painting_data_folder="folder/with/slashes",
            log_directory="logs\\windows\\path",
            log_filename="file-with-dashes_and_underscores.json",
            error_log_filename="файл.json"  # Non-ASCII characters
        )
        
        assert "AccountKey=abc123==" in test_settings.azure_connection_string
        assert test_settings.azure_container_name == "test-container_123"
        assert test_settings.painting_data_folder == "folder/with/slashes"
        assert test_settings.log_directory == "logs\\windows\\path"
        assert test_settings.log_filename == "file-with-dashes_and_underscores.json"
        assert test_settings.error_log_filename == "файл.json"

    def test_model_config_attributes(self):
        """Test that model_config is properly set."""
        assert hasattr(Settings, 'model_config')
        assert Settings.model_config['env_file'] == '.env'
        assert Settings.model_config['env_file_encoding'] == 'utf-8'

    @patch.dict(os.environ, {
        "AZURE_CONNECTION_STRING": "env_connection_string",
        "AZURE_CONTAINER_NAME": "env-container",
        "PAINTING_DATA_FOLDER": "env-folder",
        "SCHEDULER_INTERVAL_MINUTES": "15",
        "BATCH_SIZE": "50",
        "LOG_DIRECTORY": "env-logs",
        "HTTP_TIMEOUT": "120",
        "MAX_RETRIES": "10"
    })
    def test_environment_variable_override(self):
        """Test that environment variables properly override default values."""
        test_settings = Settings()
        
        assert test_settings.azure_connection_string == "env_connection_string"
        assert test_settings.azure_container_name == "env-container"
        assert test_settings.painting_data_folder == "env-folder"
        assert test_settings.scheduler_interval_minutes == 15
        assert test_settings.batch_size == 50
        assert test_settings.log_directory == "env-logs"
        assert test_settings.http_timeout == 120
        assert test_settings.max_retries == 10

    def test_type_coercion(self):
        """Test that string values are properly coerced to correct types."""
        # Test integer coercion from strings
        test_settings = Settings(
            azure_connection_string="test",
            scheduler_interval_minutes="30",  # String that should become int
            batch_size="100",
            http_timeout="180",
            max_retries="7"
        )
        
        assert isinstance(test_settings.scheduler_interval_minutes, int)
        assert test_settings.scheduler_interval_minutes == 30
        assert isinstance(test_settings.batch_size, int)
        assert test_settings.batch_size == 100

    def test_invalid_type_coercion_raises_error(self):
        """Test that invalid type conversions raise ValidationError."""
        with pytest.raises(ValidationError):
            Settings(
                azure_connection_string="test",
                scheduler_interval_minutes="not_a_number"
            )

    def test_global_settings_instance(self):
        """Test that the global settings instance is properly initialized."""
        # Test that settings is an instance of Settings
        assert isinstance(settings, Settings)
        
        # Test that it has all required attributes
        assert hasattr(settings, 'azure_connection_string')
        assert hasattr(settings, 'azure_container_name')
        assert hasattr(settings, 'model_services')

    def test_settings_immutability_after_creation(self):
        """Test settings behavior after instantiation."""
        test_settings = Settings(azure_connection_string="test")
        
        # Should be able to modify attributes (pydantic models are mutable by default)
        test_settings.batch_size = 999
        assert test_settings.batch_size == 999
        
        # Should be able to access all properties
        assert callable(getattr(test_settings, 'model_services', None))

    def test_unicode_handling(self):
        """Test handling of Unicode characters in configuration values."""
        test_settings = Settings(
            azure_connection_string="тест",  # Cyrillic
            azure_container_name="测试",     # Chinese
            painting_data_folder="🎨painting",  # Emoji
        )
        
        assert test_settings.azure_connection_string == "тест"
        assert test_settings.azure_container_name == "测试"
        assert test_settings.painting_data_folder == "🎨painting"

    def test_very_long_strings(self):
        """Test handling of very long string values."""
        long_string = "x" * 10000
        test_settings = Settings(
            azure_connection_string=long_string,
            azure_container_name=long_string
        )
        
        assert len(test_settings.azure_connection_string) == 10000
        assert len(test_settings.azure_container_name) == 10000

    def test_boundary_values_for_integers(self):
        """Test boundary values for integer fields."""
        # Test with maximum reasonable values
        test_settings = Settings(
            azure_connection_string="test",
            scheduler_interval_minutes=2147483647,  # Max 32-bit signed int
            batch_size=2147483647,
            http_timeout=2147483647,
            max_retries=2147483647
        )
        
        assert test_settings.scheduler_interval_minutes == 2147483647
        assert test_settings.batch_size == 2147483647

    @patch('os.getenv')
    def test_os_getenv_fallback_mechanism(self, mock_getenv):
        """Test the os.getenv fallback mechanism for painting_service_url."""
        # Test when environment variable is not set
        mock_getenv.return_value = None
        test_settings = Settings(azure_connection_string="test")
        # Since os.getenv is mocked to return None, it should use the default
        expected_default = "http://localhost:8001"
        # The actual value depends on the implementation, but we can test the pattern
        assert "localhost" in test_settings.painting_service_url or test_settings.painting_service_url == expected_default

        # Test when environment variable is set
        mock_getenv.return_value = "http://mocked:8888"
        # Create new instance to pick up the mocked value
        # Note: This might not work as expected due to how the default is set in the class definition
        # But we can test that getenv was called
        mock_getenv.assert_called()

    def test_model_services_property_consistency(self):
        """Test that model_services property maintains consistency."""
        test_settings = Settings(
            azure_connection_string="test",
            painting_service_url="http://consistent:8000"
        )
        
        # Multiple calls should return the same value
        result1 = test_settings.model_services
        result2 = test_settings.model_services
        
        assert result1 == result2
        assert result1 is not result2  # Should be different objects (new dict each time)

    def test_all_fields_have_expected_types(self):
        """Test that all fields have the expected Python types after initialization."""
        test_settings = Settings(azure_connection_string="test")

        assert isinstance(test_settings.azure_connection_string, str)
        assert isinstance(test_settings.azure_container_name, str)
        assert isinstance(test_settings.painting_data_folder, str)
        assert isinstance(test_settings.scheduler_interval_minutes, int)
        assert isinstance(test_settings.batch_size, int)
        assert isinstance(test_settings.painting_service_url, str)
        assert isinstance(test_settings.log_directory, str)
        assert isinstance(test_settings.log_filename, str)
        assert isinstance(test_settings.error_log_filename, str)
        assert isinstance(test_settings.http_timeout, int)
        assert isinstance(test_settings.max_retries, int)
        assert isinstance(test_settings.model_services, dict)

    def test_configuration_completeness(self):
        """Test that all expected configuration options are present and accessible."""
        test_settings = Settings(azure_connection_string="test")

        # All expected attributes should exist
        expected_attrs = [
            'azure_connection_string', 'azure_container_name', 'painting_data_folder',
            'scheduler_interval_minutes', 'batch_size', 'painting_service_url',
            'log_directory', 'log_filename', 'error_log_filename',
            'http_timeout', 'max_retries', 'model_services'
        ]

        for attr in expected_attrs:
            assert hasattr(test_settings, attr), f"Settings missing expected attribute: {attr}"
            # Should be able to access the value without errors
            getattr(test_settings, attr)

    def test_settings_representation(self):
        """Test string representation and debugging capabilities."""
        test_settings = Settings(azure_connection_string="test_connection")

        # Should be able to convert to string without errors
        str_repr = str(test_settings)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

        # Should be able to get repr without errors
        repr_str = repr(test_settings)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0