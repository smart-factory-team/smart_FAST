import json
import os
import tempfile
import pytest
from unittest.mock import patch, mock_open, call
import shutil

# Import the logger module from the correct path
from app.utils.logger import AnomalyLogger, anomaly_logger


class TestAnomalyLogger:
    """
    Comprehensive unit tests for AnomalyLogger class.
    Testing framework: pytest (as identified in existing project structure)
    """

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock settings to avoid dependency on actual config

        self.settings_patcher = patch('app.utils.logger.settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.log_directory = '/tmp/test_logs'
        self.mock_settings.log_filename = 'anomaly_test.log'
        self.mock_settings.error_log_filename = 'error_test.log'

        # Create temporary directory for tests

        self.temp_dir = tempfile.mkdtemp()
        self.mock_settings.log_directory = self.temp_dir

        yield

        # Clean up after test
        self.settings_patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('app.utils.logger.os.makedirs')
    def test_init_creates_log_directory(self, mock_makedirs):
        """Test that AnomalyLogger initialization creates log directory."""
        with patch('app.utils.logger.settings') as mock_settings:
            mock_settings.log_directory = '/test/logs'
            mock_settings.log_filename = 'test.log'
            AnomalyLogger()
            mock_makedirs.assert_called_once_with('/test/logs', exist_ok=True)

    def test_init_sets_correct_log_file_path(self):
        """Test that AnomalyLogger sets the correct log file path."""
        logger = AnomalyLogger()
        expected_path = os.path.join(self.temp_dir, 'anomaly_test.log')
        assert logger.log_file_path == expected_path

    @patch('app.utils.logger.datetime')
    @patch('builtins.print')
    @patch('builtins.open', new_callable=mock_open)
    def test_log_anomaly_writes_correct_json_format(self, mock_file, mock_print, mock_datetime):
        """Test that log_anomaly writes the correct JSON format to file."""
        # Setup mock datetime
        mock_datetime.now.return_value.isoformat.return_value = '2023-12-01T10:30:00'

        # Test data
        service_name = "TestService"
        prediction_result = {
            'machineId': 'MACHINE_001',
            'timeStamp': '2023-12-01T10:30:00',
            'issue': 'Temperature anomaly'
        }
        original_data = {'sensor_data': {'temperature': 85.5}}

        logger = AnomalyLogger()

        # Execute
        logger.log_anomaly(service_name, prediction_result, original_data)

        # Verify file operations
        expected_log_path = os.path.join(self.temp_dir, 'anomaly_test.log')
        mock_file.assert_called_once_with(expected_log_path, "a", encoding="utf-8")

        # Verify JSON content written
        handle = mock_file.return_value.__enter__.return_value
        written_content = handle.write.call_args[0][0]

        # Parse and verify the JSON content
        json_part = written_content.rstrip('\n')
        parsed_json = json.loads(json_part)

        expected_json = {
            "timestamp": "2023-12-01T10:30:00",
            "service_name": service_name,
            "prediction": prediction_result,
            "original_data": original_data
        }

        assert parsed_json == expected_json

    @patch('builtins.print')
    def test_log_anomaly_console_output(self, mock_print):
        """Test that log_anomaly produces correct console output."""
        service_name = "TestService"
        prediction_result = {
            'machineId': 'MACHINE_001',
            'timeStamp': '2023-12-01T10:30:00',
            'issue': 'Temperature anomaly'
        }
        original_data = {'sensor_data': {'temperature': 85.5}}

        logger = AnomalyLogger()

        with patch('builtins.open', mock_open()):
            logger.log_anomaly(service_name, prediction_result, original_data)

        # Verify console output
        expected_calls = [
            call("🚨 ANOMALY DETECTED: TestService"),
            call("   └─ Machine ID: MACHINE_001"),
            call("   └─ Time: 2023-12-01T10:30:00"),
            call("   └─ Issue: Temperature anomaly"),
            call("-" * 50)
        ]

        mock_print.assert_has_calls(expected_calls)

    @patch('builtins.print')
    def test_log_anomaly_console_output_with_missing_fields(self, mock_print):
        """Test console output when prediction_result has missing fields."""
        service_name = "TestService"
        prediction_result = {}  # Empty prediction result
        original_data = {'sensor_data': {'temperature': 85.5}}

        logger = AnomalyLogger()

        with patch('builtins.open', mock_open()):
            logger.log_anomaly(service_name, prediction_result, original_data)

        # Verify console output handles missing fields gracefully
        expected_calls = [
            call("🚨 ANOMALY DETECTED: TestService"),
            call("   └─ Machine ID: N/A"),
            call("   └─ Time: N/A"),
            call("   └─ Issue: N/A"),
            call("-" * 50)
        ]

        mock_print.assert_has_calls(expected_calls)

    @patch('builtins.print')
    def test_log_normal_processing(self, mock_print):
        """Test log_normal_processing console output."""
        service_name = "TestService"
        original_data = {
            'machineId': 'MACHINE_001',
            'timeStamp': '2023-12-01T10:30:00'
        }

        logger = AnomalyLogger()
        logger.log_normal_processing(service_name, original_data)

        expected_message = "✅ NORMAL: TestService - Machine ID: MACHINE_001, Time: 2023-12-01T10:30:00"
        mock_print.assert_called_once_with(expected_message)

    @patch('builtins.print')
    def test_log_normal_processing_with_missing_fields(self, mock_print):
        """Test log_normal_processing handles missing fields in original_data."""
        service_name = "TestService"
        original_data = {}  # Empty data

        logger = AnomalyLogger()
        logger.log_normal_processing(service_name, original_data)

        expected_message = "✅ NORMAL: TestService - Machine ID: N/A, Time: N/A"
        mock_print.assert_called_once_with(expected_message)

    @patch('app.utils.logger.datetime')
    @patch('builtins.print')
    @patch('builtins.open', new_callable=mock_open)
    def test_log_error_with_original_data(self, mock_file, mock_print, mock_datetime):
        """Test log_error writes error to file and prints to console."""
        # Setup mock datetime
        mock_datetime.now.return_value.isoformat.return_value = '2023-12-01T10:30:00'

        service_name = "TestService"
        error_message = "Connection timeout"
        original_data = {'machineId': 'MACHINE_001'}

        logger = AnomalyLogger()
        logger.log_error(service_name, error_message, original_data)

        # Verify file operations
        expected_error_log_path = os.path.join(self.temp_dir, 'error_test.log')
        mock_file.assert_called_once_with(expected_error_log_path, "a", encoding="utf-8")

        # Verify JSON content
        handle = mock_file.return_value.__enter__.return_value
        written_content = handle.write.call_args[0][0]
        json_part = written_content.rstrip('\n')
        parsed_json = json.loads(json_part)

        expected_json = {
            "timestamp": "2023-12-01T10:30:00",
            "service_name": service_name,
            "error": error_message,
            "original_data": original_data
        }

        assert parsed_json == expected_json

        # Verify console output
        mock_print.assert_called_once_with("❌ ERROR: TestService - Connection timeout")

    @patch('app.utils.logger.datetime')
    @patch('builtins.print')
    @patch('builtins.open', new_callable=mock_open)
    def test_log_error_without_original_data(self, mock_file, mock_print, mock_datetime):
        """Test log_error works when original_data is None."""
        mock_datetime.now.return_value.isoformat.return_value = '2023-12-01T10:30:00'

        service_name = "TestService"
        error_message = "Invalid configuration"

        logger = AnomalyLogger()
        logger.log_error(service_name, error_message)

        # Verify JSON content includes None for original_data
        handle = mock_file.return_value.__enter__.return_value
        written_content = handle.write.call_args[0][0]
        json_part = written_content.rstrip('\n')
        parsed_json = json.loads(json_part)

        assert parsed_json['original_data'] is None

    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    @patch('builtins.print')
    def test_log_anomaly_file_permission_error(self, mock_print, mock_open_func):
        """Test that file permission errors are handled appropriately."""
        service_name = "TestService"
        prediction_result = {'machineId': 'MACHINE_001'}
        original_data = {'sensor_data': {'temperature': 85.5}}

        logger = AnomalyLogger()

        with pytest.raises(PermissionError):
            logger.log_anomaly(service_name, prediction_result, original_data)

    @patch('builtins.open', side_effect=OSError("Disk full"))
    @patch('builtins.print')
    def test_log_error_disk_full_error(self, mock_print, mock_open_func):
        """Test that disk full errors are handled appropriately."""
        service_name = "TestService"
        error_message = "Connection timeout"

        logger = AnomalyLogger()

        with pytest.raises(OSError):
            logger.log_error(service_name, error_message)

    @patch('builtins.print')
    def test_log_methods_with_unicode_characters(self, mock_print):
        """Test that all log methods handle Unicode characters correctly."""
        service_name = "서비스테스트"  # Korean characters

        # Test log_normal_processing with Unicode
        original_data = {
            'machineId': 'MACHINE_001',
            'timeStamp': '2023-12-01T10:30:00',
            'description': '온도 센서'  # Korean characters
        }

        logger = AnomalyLogger()
        logger.log_normal_processing(service_name, original_data)

        # Verify it doesn't raise encoding errors
        assert mock_print.called

    @patch('app.utils.logger.datetime')
    @patch('builtins.open', new_callable=mock_open)
    def test_json_serialization_with_complex_data_types(self, mock_file, mock_datetime):
        """Test JSON serialization handles complex data types correctly."""
        mock_datetime.now.return_value.isoformat.return_value = '2023-12-01T10:30:00'

        # Test with nested dictionaries and lists
        service_name = "TestService"
        prediction_result = {
            'machineId': 'MACHINE_001',
            'nested_data': {
                'sensors': ['temp', 'pressure'],
                'readings': {'temp': 25.5, 'pressure': 1.2}
            }
        }
        original_data = {
            'raw_sensors': [1, 2, 3, 4, 5],
            'metadata': {'source': 'sensor_array_1'}
        }

        logger = AnomalyLogger()

        with patch('builtins.print'):
            logger.log_anomaly(service_name, prediction_result, original_data)

        # Verify that complex data structures are serialized correctly
        handle = mock_file.return_value.__enter__.return_value
        written_content = handle.write.call_args[0][0]
        json_part = written_content.rstrip('\n')

        # Should not raise JSON serialization errors
        parsed_json = json.loads(json_part)
        assert isinstance(parsed_json, dict)
        assert parsed_json['service_name'] == service_name

    def test_global_logger_instance_exists(self):
        """Test that global anomaly_logger instance is created."""
        assert isinstance(anomaly_logger, AnomalyLogger)

    @patch('builtins.open', new_callable=mock_open)
    def test_multiple_anomaly_logs_append_correctly(self, mock_file):
        """Test that multiple anomaly logs are appended to the same file."""
        service_name = "TestService"
        prediction_result_1 = {'machineId': 'MACHINE_001', 'issue': 'Issue 1'}
        prediction_result_2 = {'machineId': 'MACHINE_002', 'issue': 'Issue 2'}
        original_data = {}

        logger = AnomalyLogger()

        with patch('builtins.print'):
            logger.log_anomaly(service_name, prediction_result_1, original_data)
            logger.log_anomaly(service_name, prediction_result_2, original_data)

        # Verify file was opened twice in append mode
        expected_log_path = os.path.join(self.temp_dir, 'anomaly_test.log')
        expected_calls = [
            call(expected_log_path, "a", encoding="utf-8"),
            call(expected_log_path, "a", encoding="utf-8")
        ]
        mock_file.assert_has_calls(expected_calls)

    @patch('builtins.open', new_callable=mock_open)
    def test_multiple_error_logs_append_correctly(self, mock_file):
        """Test that multiple error logs are appended to the same file."""
        service_name = "TestService"

        logger = AnomalyLogger()

        with patch('builtins.print'):
            logger.log_error(service_name, "Error 1")
            logger.log_error(service_name, "Error 2")

        # Verify error file was opened twice in append mode
        expected_error_log_path = os.path.join(self.temp_dir, 'error_test.log')
        expected_calls = [
            call(expected_error_log_path, "a", encoding="utf-8"),
            call(expected_error_log_path, "a", encoding="utf-8")
        ]
        mock_file.assert_has_calls(expected_calls)

    def test_log_file_paths_are_constructed_correctly(self):
        """Test that log file paths are constructed correctly from settings."""
        logger = AnomalyLogger()
        expected_anomaly_path = os.path.join(self.temp_dir, 'anomaly_test.log')
        assert logger.log_file_path == expected_anomaly_path

    @patch('builtins.print')
    def test_empty_service_name_handling(self, mock_print):
        """Test handling of empty service names."""
        service_name = ""
        original_data = {'machineId': 'MACHINE_001'}

        logger = AnomalyLogger()
        logger.log_normal_processing(service_name, original_data)

        expected_message = "✅ NORMAL:  - Machine ID: MACHINE_001, Time: N/A"
        mock_print.assert_called_once_with(expected_message)

    @patch('builtins.print')
    def test_none_values_handling(self, mock_print):
        """Test handling of None values in data structures."""
        service_name = "TestService"
        original_data = {
            'machineId': None,
            'timeStamp': None
        }

        logger = AnomalyLogger()
        logger.log_normal_processing(service_name, original_data)

        expected_message = "✅ NORMAL: TestService - Machine ID: None, Time: None"
        mock_print.assert_called_once_with(expected_message)

    @pytest.mark.parametrize("service_name,prediction_result,original_data", [
        ("Service1", {'machineId': 'M1'}, {'data': 'test1'}),
        ("Service2", {'machineId': 'M2', 'issue': 'Critical'}, {'data': 'test2'}),
        ("Service3", {}, {}),
    ])
    @patch('builtins.print')
    def test_log_anomaly_parametrized(self, mock_print, service_name, prediction_result, original_data):
        """Parametrized test for log_anomaly with different data combinations."""
        logger = AnomalyLogger()

        with patch('builtins.open', mock_open()):
            logger.log_anomaly(service_name, prediction_result, original_data)

        # Verify the anomaly detection message is always printed
        calls = mock_print.call_args_list
        assert any(f"🚨 ANOMALY DETECTED: {service_name}" in str(call) for call in calls)

    @patch('builtins.print')
    def test_log_normal_processing_with_special_characters(self, mock_print):
        """Test log_normal_processing with special characters in data."""
        service_name = "Test@Service#123"
        original_data = {
            'machineId': 'MACHINE-001_TEST',
            'timeStamp': '2023-12-01T10:30:00+00:00'
        }

        logger = AnomalyLogger()
        logger.log_normal_processing(service_name, original_data)

        expected_message = "✅ NORMAL: Test@Service#123 - Machine ID: MACHINE-001_TEST, Time: 2023-12-01T10:30:00+00:00"
        mock_print.assert_called_once_with(expected_message)

    @patch('app.utils.logger.datetime')
    @patch('builtins.open', new_callable=mock_open)
    def test_timestamp_format_consistency(self, mock_file, mock_datetime):
        """Test that timestamps are consistently formatted across all log methods."""
        mock_datetime.now.return_value.isoformat.return_value = '2023-12-01T10:30:00.123456'

        logger = AnomalyLogger()
        service_name = "TestService"

        with patch('builtins.print'):
            # Test anomaly log timestamp
            logger.log_anomaly(service_name, {'machineId': 'M1'}, {})

            # Test error log timestamp
            logger.log_error(service_name, "Test error")

        # Verify both calls used the same timestamp format
        calls = mock_file.return_value.__enter__.return_value.write.call_args_list

        for call_args in calls:
            written_content = call_args[0][0]
            json_part = written_content.rstrip('\n')
            parsed_json = json.loads(json_part)
            assert parsed_json['timestamp'] == '2023-12-01T10:30:00.123456'