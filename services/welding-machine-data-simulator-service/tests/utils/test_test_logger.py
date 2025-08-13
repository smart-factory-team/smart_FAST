from app.utils.logger import AnomalyLogger, anomaly_logger
import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


class TestAnomalyLogger(unittest.TestCase):
    """Test cases for AnomalyLogger class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_settings = MagicMock()
        self.mock_settings.log_directory = self.temp_dir
        self.mock_settings.log_filename = "anomaly_test.log"
        self.mock_settings.error_log_filename = "error_test.log"

        # Sample test data
        self.sample_prediction_result = {
            "mae": 0.75,
            "threshold": 0.5,
            "status": "anomaly",
            "confidence": 0.95
        }

        self.sample_original_data = {
            "sensor_1": 25.5,
            "sensor_2": 100.3,
            "timestamp": "2024-01-01T10:00:00"
        }

        self.service_name = "test_service"

    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('app.utils.logger.settings')
    def test_init_creates_log_directory(self, mock_settings):
        """Test that AnomalyLogger creates log directory on initialization"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "test.log"

        logger = AnomalyLogger()

        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(logger.log_file_path,
                         os.path.join(self.temp_dir, "test.log"))

    @patch('app.utils.logger.settings')
    @patch('app.utils.logger.datetime')
    @patch('builtins.print')
    def test_log_anomaly_writes_to_file_and_console(self, mock_print, mock_datetime, mock_settings):
        """Test that log_anomaly writes to file and prints to console"""
        # Setup mocks
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "anomaly.log"
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time

        logger = AnomalyLogger()

        # Execute
        logger.log_anomaly(
            self.service_name, self.sample_prediction_result, self.sample_original_data)

        # Verify file was written
        log_file = os.path.join(self.temp_dir, "anomaly.log")
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            log_entry = json.loads(log_content.strip())

        # Verify log entry structure
        self.assertEqual(log_entry["service_name"], self.service_name)
        self.assertEqual(log_entry["prediction"],
                         self.sample_prediction_result)
        self.assertEqual(log_entry["original_data"], self.sample_original_data)
        self.assertEqual(log_entry["timestamp"], fixed_time.isoformat())

        # Verify console output
        expected_calls = [
            unittest.mock.call(f"🚨 ANOMALY DETECTED: {self.service_name}"),
            unittest.mock.call(
                f"   └─ MAE: {self.sample_prediction_result['mae']}"),
            unittest.mock.call(
                f"   └─ Threshold: {self.sample_prediction_result['threshold']}"),
            unittest.mock.call(
                f"   └─ Status: {self.sample_prediction_result['status']}"),
            unittest.mock.call(f"   └─ Time: {fixed_time.isoformat()}"),
            unittest.mock.call("-" * 50)
        ]
        mock_print.assert_has_calls(expected_calls)

    @patch('app.utils.logger.settings')
    @patch('builtins.print')
    def test_log_anomaly_handles_missing_prediction_fields(self, mock_print, mock_settings):
        """Test that log_anomaly handles missing fields in prediction_result gracefully"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "anomaly.log"

        logger = AnomalyLogger()
        incomplete_prediction = {"mae": 0.8}  # Missing threshold and status

        logger.log_anomaly(self.service_name,
                           incomplete_prediction, self.sample_original_data)

        # Verify it doesn't crash and uses 'N/A' for missing fields
        expected_calls = [
            unittest.mock.call(f"🚨 ANOMALY DETECTED: {self.service_name}"),
            unittest.mock.call("   └─ MAE: 0.8"),
            unittest.mock.call("   └─ Threshold: N/A"),
            unittest.mock.call("   └─ Status: N/A"),
        ]

        for call in expected_calls:
            self.assertIn(call, mock_print.call_args_list)

    @patch('app.utils.logger.settings')
    @patch('builtins.print')
    def test_log_normal_processing_console_only(self, mock_print, mock_settings):
        """Test that log_normal_processing only prints to console, not file"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "anomaly.log"

        logger = AnomalyLogger()

        logger.log_normal_processing(
            self.service_name, self.sample_prediction_result)

        # Verify console output
        expected_message = f"✅ NORMAL: {self.service_name} - MAE: {self.sample_prediction_result['mae']}"
        mock_print.assert_called_with(expected_message)

        # Verify no file was created
        log_file = os.path.join(self.temp_dir, "anomaly.log")
        self.assertFalse(os.path.exists(log_file))

    @patch('app.utils.logger.settings')
    @patch('builtins.print')
    def test_log_normal_processing_handles_missing_mae(self, mock_print, mock_settings):
        """Test that log_normal_processing handles missing MAE field"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "anomaly.log"

        logger = AnomalyLogger()
        prediction_without_mae = {"threshold": 0.5, "status": "normal"}

        logger.log_normal_processing(self.service_name, prediction_without_mae)

        expected_message = f"✅ NORMAL: {self.service_name} - MAE: N/A"
        mock_print.assert_called_with(expected_message)

    @patch('app.utils.logger.settings')
    @patch('app.utils.logger.datetime')
    @patch('builtins.print')
    def test_log_error_writes_to_error_file(self, mock_print, mock_datetime, mock_settings):
        """Test that log_error writes to error log file and prints to console"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.error_log_filename = "error.log"
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time

        logger = AnomalyLogger()
        error_message = "Test error occurred"

        logger.log_error(self.service_name, error_message,
                         self.sample_original_data)

        # Verify error file was written
        error_file = os.path.join(self.temp_dir, "error.log")
        self.assertTrue(os.path.exists(error_file))

        with open(error_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            log_entry = json.loads(log_content.strip())

        # Verify log entry structure
        self.assertEqual(log_entry["service_name"], self.service_name)
        self.assertEqual(log_entry["error"], error_message)
        self.assertEqual(log_entry["original_data"], self.sample_original_data)
        self.assertEqual(log_entry["timestamp"], fixed_time.isoformat())

        # Verify console output
        expected_message = f"❌ ERROR: {self.service_name} - {error_message}"
        mock_print.assert_called_with(expected_message)

    @patch('app.utils.logger.settings')
    @patch('builtins.print')
    def test_log_error_without_original_data(self, mock_print, mock_settings):
        """Test that log_error works when original_data is None"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.error_log_filename = "error.log"

        logger = AnomalyLogger()
        error_message = "Test error without data"

        logger.log_error(self.service_name, error_message)

        # Verify error file was written
        error_file = os.path.join(self.temp_dir, "error.log")
        self.assertTrue(os.path.exists(error_file))

        with open(error_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            log_entry = json.loads(log_content.strip())

        # Verify original_data is None
        self.assertIsNone(log_entry["original_data"])

    @patch('app.utils.logger.settings')
    def test_multiple_log_entries_append_correctly(self, mock_settings):
        """Test that multiple log entries are appended correctly to the same file"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "anomaly.log"

        logger = AnomalyLogger()

        # Write multiple entries
        for i in range(3):
            prediction = {"mae": i * 0.1, "status": f"test_{i}"}
            original_data = {"value": i}
            logger.log_anomaly(f"service_{i}", prediction, original_data)

        # Verify all entries are in the file
        log_file = os.path.join(self.temp_dir, "anomaly.log")
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)

        for i, line in enumerate(lines):
            log_entry = json.loads(line.strip())
            self.assertEqual(log_entry["service_name"], f"service_{i}")
            self.assertEqual(log_entry["prediction"]["mae"], i * 0.1)

    @patch('app.utils.logger.settings')
    def test_unicode_handling_in_logs(self, mock_settings):
        """Test that logger handles unicode characters correctly"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "anomaly.log"

        logger = AnomalyLogger()

        # Test with unicode characters
        unicode_service = "서비스_테스트"
        unicode_data = {"message": "한글 메시지", "emoji": "🚨"}

        logger.log_anomaly(
            unicode_service, self.sample_prediction_result, unicode_data)

        # Verify unicode is preserved
        log_file = os.path.join(self.temp_dir, "anomaly.log")
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            log_entry = json.loads(log_content.strip())

        self.assertEqual(log_entry["service_name"], unicode_service)
        self.assertEqual(log_entry["original_data"]["message"], "한글 메시지")
        self.assertEqual(log_entry["original_data"]["emoji"], "🚨")

    @patch('app.utils.logger.settings')
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_file_permission_error_handling(self, mock_open, mock_settings):
        """Test behavior when file permissions prevent writing"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "anomaly.log"

        logger = AnomalyLogger()

        # This should raise an exception since we can't handle the permission error
        with self.assertRaises(PermissionError):
            logger.log_anomaly(
                self.service_name, self.sample_prediction_result, self.sample_original_data)

    @patch('app.utils.logger.settings')
    def test_empty_prediction_result(self, mock_settings):
        """Test logging with empty prediction result"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "anomaly.log"

        logger = AnomalyLogger()
        empty_prediction = {}

        # Should not crash
        logger.log_anomaly(self.service_name, empty_prediction,
                           self.sample_original_data)

        log_file = os.path.join(self.temp_dir, "anomaly.log")
        with open(log_file, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.read().strip())

        self.assertEqual(log_entry["prediction"], {})

    @patch('app.utils.logger.settings')
    def test_large_data_serialization(self, mock_settings):
        """Test logging with large data structures"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "anomaly.log"

        logger = AnomalyLogger()

        # Create large data structure
        large_data = {f"sensor_{i}": i * 0.1 for i in range(1000)}
        large_prediction = {
            "mae": 0.9,
            "detailed_results": [{"index": i, "value": i} for i in range(100)]
        }

        logger.log_anomaly(self.service_name, large_prediction, large_data)

        # Verify it was written correctly
        log_file = os.path.join(self.temp_dir, "anomaly.log")
        self.assertTrue(os.path.exists(log_file))

        with open(log_file, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.read().strip())

        self.assertEqual(len(log_entry["original_data"]), 1000)
        self.assertEqual(len(log_entry["prediction"]["detailed_results"]), 100)


class TestGlobalAnomalyLoggerInstance(unittest.TestCase):
    """Test cases for the global anomaly_logger instance"""

    def test_global_logger_instance_exists(self):
        """Test that global anomaly_logger instance is created"""
        self.assertIsInstance(anomaly_logger, AnomalyLogger)
        self.assertIsNotNone(anomaly_logger.log_file_path)

    @patch('app.utils.logger.settings')
    def test_global_logger_methods_callable(self, mock_settings):
        """Test that global logger instance methods are callable"""
        mock_settings.log_directory = tempfile.mkdtemp()
        mock_settings.log_filename = "test.log"
        mock_settings.error_log_filename = "error.log"

        # These should not raise exceptions
        self.assertTrue(hasattr(anomaly_logger, 'log_anomaly'))
        self.assertTrue(hasattr(anomaly_logger, 'log_normal_processing'))
        self.assertTrue(hasattr(anomaly_logger, 'log_error'))
        self.assertTrue(callable(anomaly_logger.log_anomaly))
        self.assertTrue(callable(anomaly_logger.log_normal_processing))
        self.assertTrue(callable(anomaly_logger.log_error))


class TestAnomalyLoggerIntegration(unittest.TestCase):
    """Integration tests for AnomalyLogger with real file operations"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('app.utils.logger.settings')
    def test_full_logging_workflow(self, mock_settings):
        """Test complete logging workflow with real file operations"""
        mock_settings.log_directory = self.temp_dir
        mock_settings.log_filename = "integration_anomaly.log"
        mock_settings.error_log_filename = "integration_error.log"

        logger = AnomalyLogger()

        # Test anomaly logging
        prediction = {"mae": 0.8, "threshold": 0.5, "status": "anomaly"}
        original_data = {"sensor_temp": 85.5, "sensor_pressure": 120.3}

        logger.log_anomaly("integration_test", prediction, original_data)

        # Test error logging
        logger.log_error("integration_test",
                         "Integration test error", original_data)

        # Verify both files exist and contain correct data
        anomaly_file = os.path.join(self.temp_dir, "integration_anomaly.log")
        error_file = os.path.join(self.temp_dir, "integration_error.log")

        self.assertTrue(os.path.exists(anomaly_file))
        self.assertTrue(os.path.exists(error_file))

        # Verify content
        with open(anomaly_file, 'r', encoding='utf-8') as f:
            anomaly_log = json.loads(f.read().strip())

        with open(error_file, 'r', encoding='utf-8') as f:
            error_log = json.loads(f.read().strip())

        self.assertEqual(anomaly_log["service_name"], "integration_test")
        self.assertEqual(error_log["service_name"], "integration_test")
        self.assertEqual(error_log["error"], "Integration test error")


if __name__ == '__main__':
    unittest.main()
