import json
import os
import tempfile
import pytest
from datetime import datetime
from unittest.mock import patch
from pydantic import ValidationError

# Import the modules we're testing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import IssueLogInput, save_issue_log


class TestIssueLogInput:
    """
    Test cases for IssueLogInput Pydantic model
    Testing Framework: pytest (as identified from existing project structure)
    """
    
    def test_valid_issue_log_input_creation(self):
        """Test creating IssueLogInput with valid data"""
        valid_data = {
            "machineId": "MACHINE_001",
            "timeStamp": datetime(2024, 1, 15, 10, 30, 45),
            "thick": 2.5,
            "voltage": 220.0,
            "current": 15.5,
            "temper": 65.2
        }
        
        issue_log = IssueLogInput(**valid_data)
        
        assert issue_log.machineId == "MACHINE_001"
        assert issue_log.timeStamp == datetime(2024, 1, 15, 10, 30, 45)
        assert issue_log.thick == 2.5
        assert issue_log.voltage == 220.0
        assert issue_log.current == 15.5
        assert issue_log.temper == 65.2

    def test_issue_log_input_with_string_timestamp(self):
        """Test IssueLogInput with ISO format timestamp string (should auto-convert)"""
        valid_data = {
            "machineId": "MACHINE_002",
            "timeStamp": "2024-01-15T10:30:45",
            "thick": 3.0,
            "voltage": 240.0,
            "current": 18.0,
            "temper": 70.5
        }
        
        issue_log = IssueLogInput(**valid_data)
        
        assert issue_log.machineId == "MACHINE_002"
        assert isinstance(issue_log.timeStamp, datetime)
        assert issue_log.timeStamp.year == 2024
        assert issue_log.timeStamp.month == 1
        assert issue_log.timeStamp.day == 15

    def test_issue_log_input_with_string_timestamp_microseconds(self):
        """Test IssueLogInput with timestamp string including microseconds"""
        valid_data = {
            "machineId": "MACHINE_003",
            "timeStamp": "2024-01-15T10:30:45.123456",
            "thick": 2.8,
            "voltage": 230.0,
            "current": 16.2,
            "temper": 68.1
        }
        
        issue_log = IssueLogInput(**valid_data)
        
        assert issue_log.machineId == "MACHINE_003"
        assert isinstance(issue_log.timeStamp, datetime)
        assert issue_log.timeStamp.microsecond == 123456

    def test_issue_log_input_missing_required_field(self):
        """Test IssueLogInput with missing required field"""
        invalid_data = {
            "machineId": "MACHINE_004",
            # Missing timeStamp
            "thick": 2.8,
            "voltage": 230.0,
            "current": 16.2,
            "temper": 68.1
        }
        
        with pytest.raises(ValidationError):
            IssueLogInput(**invalid_data)

    def test_issue_log_input_invalid_types(self):
        """Test IssueLogInput with invalid data types"""
        invalid_data = {
            "machineId": "MACHINE_005",
            "timeStamp": datetime(2024, 1, 15, 10, 30, 45),
            "thick": "invalid_float",  # Should be float
            "voltage": 220.0,
            "current": 15.5,
            "temper": 65.2
        }
        
        with pytest.raises(ValidationError):
            IssueLogInput(**invalid_data)

    def test_issue_log_input_invalid_timestamp_format(self):
        """Test IssueLogInput with invalid timestamp format"""
        invalid_data = {
            "machineId": "MACHINE_006",
            "timeStamp": "invalid-timestamp",
            "thick": 2.5,
            "voltage": 220.0,
            "current": 15.5,
            "temper": 65.2
        }
        
        with pytest.raises(ValidationError):
            IssueLogInput(**invalid_data)

    def test_issue_log_input_negative_values(self):
        """Test IssueLogInput with negative numeric values (edge case)"""
        edge_case_data = {
            "machineId": "MACHINE_007",
            "timeStamp": datetime(2024, 1, 15, 10, 30, 45),
            "thick": -1.0,
            "voltage": -220.0,
            "current": -15.5,
            "temper": -10.0
        }
        
        # Assuming negative values are technically valid for the model
        # (business logic validation would be separate)
        issue_log = IssueLogInput(**edge_case_data)
        
        assert issue_log.thick == -1.0
        assert issue_log.voltage == -220.0
        assert issue_log.current == -15.5
        assert issue_log.temper == -10.0

    def test_issue_log_input_zero_values(self):
        """Test IssueLogInput with zero values"""
        zero_data = {
            "machineId": "MACHINE_008",
            "timeStamp": datetime(2024, 1, 15, 10, 30, 45),
            "thick": 0.0,
            "voltage": 0.0,
            "current": 0.0,
            "temper": 0.0
        }
        
        issue_log = IssueLogInput(**zero_data)
        
        assert issue_log.thick == 0.0
        assert issue_log.voltage == 0.0
        assert issue_log.current == 0.0
        assert issue_log.temper == 0.0

    def test_issue_log_input_extreme_values(self):
        """Test IssueLogInput with extreme numeric values"""
        extreme_data = {
            "machineId": "MACHINE_009",
            "timeStamp": datetime(2024, 1, 15, 10, 30, 45),
            "thick": 999999.99,
            "voltage": 999999.99,
            "current": 999999.99,
            "temper": 999999.99
        }
        
        issue_log = IssueLogInput(**extreme_data)
        
        assert issue_log.thick == 999999.99
        assert issue_log.voltage == 999999.99
        assert issue_log.current == 999999.99
        assert issue_log.temper == 999999.99

    def test_issue_log_input_empty_machine_id(self):
        """Test IssueLogInput with empty machine ID"""
        data_with_empty_id = {
            "machineId": "",
            "timeStamp": datetime(2024, 1, 15, 10, 30, 45),
            "thick": 2.5,
            "voltage": 220.0,
            "current": 15.5,
            "temper": 65.2
        }
        
        # Empty string should be valid for machineId (business logic can validate further)
        issue_log = IssueLogInput(**data_with_empty_id)
        assert issue_log.machineId == ""

    def test_issue_log_input_very_long_machine_id(self):
        """Test IssueLogInput with very long machine ID"""
        long_machine_id = "A" * 1000  # 1000 character machine ID
        data_with_long_id = {
            "machineId": long_machine_id,
            "timeStamp": datetime(2024, 1, 15, 10, 30, 45),
            "thick": 2.5,
            "voltage": 220.0,
            "current": 15.5,
            "temper": 65.2
        }
        
        issue_log = IssueLogInput(**data_with_long_id)
        assert issue_log.machineId == long_machine_id
        assert len(issue_log.machineId) == 1000

    def test_issue_log_input_special_characters_in_machine_id(self):
        """Test IssueLogInput with special characters in machine ID"""
        special_machine_id = "MACHINE-001_테스트@#$%^&*()"
        data_with_special_id = {
            "machineId": special_machine_id,
            "timeStamp": datetime(2024, 1, 15, 10, 30, 45),
            "thick": 2.5,
            "voltage": 220.0,
            "current": 15.5,
            "temper": 65.2
        }
        
        issue_log = IssueLogInput(**data_with_special_id)
        assert issue_log.machineId == special_machine_id

    def test_issue_log_input_precision_float_values(self):
        """Test IssueLogInput with high precision float values"""
        precision_data = {
            "machineId": "MACHINE_010",
            "timeStamp": datetime(2024, 1, 15, 10, 30, 45),
            "thick": 2.123456789,
            "voltage": 220.987654321,
            "current": 15.555555555,
            "temper": 65.123456789
        }
        
        issue_log = IssueLogInput(**precision_data)
        
        # Float precision may be limited, but should be close
        assert abs(issue_log.thick - 2.123456789) < 1e-9
        assert abs(issue_log.voltage - 220.987654321) < 1e-9
        assert abs(issue_log.current - 15.555555555) < 1e-9
        assert abs(issue_log.temper - 65.123456789) < 1e-9


class TestSaveIssueLog:
    """
    Test cases for save_issue_log function
    Testing Framework: pytest (as identified from existing project structure)
    """
    
    def test_save_issue_log_success(self):
        """Test successful log saving with valid config and data"""
        log_data = {
            "machineId": "MACHINE_001",
            "issue": "High temperature detected",
            "timestamp": "2024-01-15T10:30:45",
            "severity": "WARNING"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "test_logs.jsonl")
            config = {
                "logs": {
                    "file_path": log_file_path
                }
            }
            
            save_issue_log(log_data, config)
            
            # Verify file was created and contains expected data
            assert os.path.exists(log_file_path)
            
            with open(log_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.loads(f.read().strip())
                assert saved_data == log_data

    def test_save_issue_log_multiple_entries(self):
        """Test saving multiple log entries (JSON Lines format)"""
        log_data_1 = {"machineId": "MACHINE_001", "issue": "Issue 1"}
        log_data_2 = {"machineId": "MACHINE_002", "issue": "Issue 2"}
        log_data_3 = {"machineId": "MACHINE_003", "issue": "Issue 3"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "multi_logs.jsonl")
            config = {
                "logs": {
                    "file_path": log_file_path
                }
            }
            
            save_issue_log(log_data_1, config)
            save_issue_log(log_data_2, config)
            save_issue_log(log_data_3, config)
            
            # Verify all entries were saved
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
                assert len(lines) == 3
                assert json.loads(lines[0]) == log_data_1
                assert json.loads(lines[1]) == log_data_2
                assert json.loads(lines[2]) == log_data_3

    def test_save_issue_log_config_none(self):
        """Test save_issue_log with None config"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        
        with patch('builtins.print') as mock_print:
            save_issue_log(log_data, None)
            mock_print.assert_called_once_with("오류: 설정이 인자로 전달되지 않아 로그를 저장할 수 없습니다.")

    def test_save_issue_log_missing_logs_config(self):
        """Test save_issue_log with config missing logs section"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        config = {"other_setting": "value"}
        
        with patch('builtins.print') as mock_print:
            save_issue_log(log_data, config)
            mock_print.assert_called_once_with("경고: 설정 파일에 로그 저장 경로(logs.file_path)가 지정되지 않았습니다. 로그를 파일에 저장하지 않습니다.")

    def test_save_issue_log_empty_logs_config(self):
        """Test save_issue_log with empty logs config"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        config = {"logs": {}}
        
        with patch('builtins.print') as mock_print:
            save_issue_log(log_data, config)
            mock_print.assert_called_once_with("경고: 설정 파일에 로그 저장 경로(logs.file_path)가 지정되지 않았습니다. 로그를 파일에 저장하지 않습니다.")

    def test_save_issue_log_missing_file_path(self):
        """Test save_issue_log with logs config but missing file_path"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        config = {"logs": {"other_setting": "value"}}
        
        with patch('builtins.print') as mock_print:
            save_issue_log(log_data, config)
            mock_print.assert_called_once_with("경고: 설정 파일에 로그 저장 경로(logs.file_path)가 지정되지 않았습니다. 로그를 파일에 저장하지 않습니다.")

    def test_save_issue_log_empty_file_path(self):
        """Test save_issue_log with empty file_path"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        config = {"logs": {"file_path": ""}}
        
        with patch('builtins.print') as mock_print:
            save_issue_log(log_data, config)
            mock_print.assert_called_once_with("경고: 설정 파일에 로그 저장 경로(logs.file_path)가 지정되지 않았습니다. 로그를 파일에 저장하지 않습니다.")

    def test_save_issue_log_none_file_path(self):
        """Test save_issue_log with None file_path"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        config = {"logs": {"file_path": None}}
        
        with patch('builtins.print') as mock_print:
            save_issue_log(log_data, config)
            mock_print.assert_called_once_with("경고: 설정 파일에 로그 저장 경로(logs.file_path)가 지정되지 않았습니다. 로그를 파일에 저장하지 않습니다.")

    def test_save_issue_log_creates_directory(self):
        """Test that save_issue_log creates directories if they don't exist"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "logs", "subdir", "app.jsonl")
            config = {"logs": {"file_path": nested_path}}
            
            # Ensure the nested directory doesn't exist initially
            assert not os.path.exists(os.path.dirname(nested_path))
            
            save_issue_log(log_data, config)
            
            # Verify directory was created and file exists
            assert os.path.exists(os.path.dirname(nested_path))
            assert os.path.exists(nested_path)

    def test_save_issue_log_creates_deeply_nested_directory(self):
        """Test that save_issue_log creates deeply nested directories"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            deep_path = os.path.join(temp_dir, "a", "b", "c", "d", "e", "logs.jsonl")
            config = {"logs": {"file_path": deep_path}}
            
            save_issue_log(log_data, config)
            
            # Verify deeply nested directory was created and file exists
            assert os.path.exists(os.path.dirname(deep_path))
            assert os.path.exists(deep_path)

    def test_save_issue_log_no_directory_creation_needed(self):
        """Test save_issue_log when file is in current directory (no dir creation needed)"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                config = {"logs": {"file_path": "direct_log.jsonl"}}
                save_issue_log(log_data, config)
                
                # Verify file was created in current directory
                assert os.path.exists("direct_log.jsonl")
            finally:
                os.chdir(original_cwd)

    @patch('builtins.open')
    def test_save_issue_log_file_write_error(self, mock_open_func):
        """Test save_issue_log handles file write errors gracefully"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        config = {"logs": {"file_path": "/some/log/path.jsonl"}}
        
        # Mock open to raise an exception
        mock_open_func.side_effect = IOError("Permission denied")
        
        with patch('builtins.print') as mock_print:
            save_issue_log(log_data, config)
            mock_print.assert_called_once_with("로그 파일 저장 중 오류 발생: Permission denied")

    @patch('builtins.open')
    def test_save_issue_log_file_write_different_errors(self, mock_open_func):
        """Test save_issue_log handles various file write errors"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        config = {"logs": {"file_path": "/some/log/path.jsonl"}}
        
        # Test different types of exceptions
        exceptions_to_test = [
            FileNotFoundError("File not found"),
            PermissionError("Permission denied"),
            OSError("Disk full"),
            Exception("Generic error")
        ]
        
        for exception in exceptions_to_test:
            mock_open_func.side_effect = exception
            
            with patch('builtins.print') as mock_print:
                save_issue_log(log_data, config)
                mock_print.assert_called_once_with(f"로그 파일 저장 중 오류 발생: {exception}")

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_save_issue_log_directory_creation_error(self, mock_open_func, mock_exists, mock_makedirs):
        """Test save_issue_log handles directory creation errors"""
        log_data = {"machineId": "MACHINE_001", "issue": "Test issue"}
        config = {"logs": {"file_path": "/restricted/logs/app.jsonl"}}
        
        # Mock directory doesn't exist and makedirs fails
        mock_exists.return_value = False
        mock_makedirs.side_effect = OSError("Permission denied")
        mock_open_func.side_effect = OSError("No such file or directory")
        
        with patch('builtins.print') as mock_print:
            save_issue_log(log_data, config)
            # Should try to write the file and fail with the file error
            mock_print.assert_called_once_with("로그 파일 저장 중 오류 발생: No such file or directory")

    def test_save_issue_log_unicode_content(self):
        """Test save_issue_log with Unicode characters"""
        log_data = {
            "machineId": "기계_001",
            "issue": "온도가 너무 높습니다",
            "description": "측정값: 온도 75°C, 전압 220V",
            "emoji": "🔥❄️⚡"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "unicode_logs.jsonl")
            config = {"logs": {"file_path": log_file_path}}
            
            save_issue_log(log_data, config)
            
            # Verify Unicode content was saved correctly
            with open(log_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.loads(f.read().strip())
                assert saved_data["machineId"] == "기계_001"
                assert saved_data["issue"] == "온도가 너무 높습니다"
                assert "75°C" in saved_data["description"]
                assert saved_data["emoji"] == "🔥❄️⚡"

    def test_save_issue_log_large_data(self):
        """Test save_issue_log with large log data"""
        # Create a log entry with large data
        large_description = "A" * 50000  # 50KB string
        log_data = {
            "machineId": "MACHINE_001",
            "issue": "Large data test",
            "description": large_description
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "large_logs.jsonl")
            config = {"logs": {"file_path": log_file_path}}
            
            save_issue_log(log_data, config)
            
            # Verify large data was saved correctly
            with open(log_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.loads(f.read().strip())
                assert len(saved_data["description"]) == 50000
                assert saved_data["description"] == large_description

    def test_save_issue_log_special_characters_in_data(self):
        """Test save_issue_log with special characters and edge case data"""
        log_data = {
            "machineId": "MACHINE_001",
            "issue": 'Special chars: "quotes", \'apostrophes\', \n newlines, \t tabs',
            "json_string": '{"nested": "json", "numbers": [1, 2, 3]}',
            "null_value": None,
            "boolean": True,
            "number": 42.5,
            "backslashes": "\\path\\to\\file",
            "forward_slashes": "/path/to/file"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "special_chars_logs.jsonl")
            config = {"logs": {"file_path": log_file_path}}
            
            save_issue_log(log_data, config)
            
            # Verify special characters were handled correctly
            with open(log_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.loads(f.read().strip())
                assert saved_data == log_data
                assert saved_data["null_value"] is None
                assert saved_data["boolean"] is True
                assert saved_data["number"] == 42.5

    def test_save_issue_log_append_mode(self):
        """Test that save_issue_log appends to existing files rather than overwriting"""
        initial_data = {"machineId": "MACHINE_001", "issue": "Initial entry"}
        new_data = {"machineId": "MACHINE_002", "issue": "New entry"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "append_test.jsonl")
            config = {"logs": {"file_path": log_file_path}}
            
            # Write initial data manually
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False)
                f.write('\n')
            
            # Use save_issue_log to append new data
            save_issue_log(new_data, config)
            
            # Verify both entries exist
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
                assert len(lines) == 2
                assert json.loads(lines[0]) == initial_data
                assert json.loads(lines[1]) == new_data

    def test_save_issue_log_complex_nested_data(self):
        """Test save_issue_log with complex nested data structures"""
        complex_log_data = {
            "machineId": "MACHINE_001",
            "measurements": {
                "temperature": [65.1, 65.3, 65.5, 65.2],
                "voltage": [220.1, 220.2, 220.0, 219.9],
                "current": [15.5, 15.6, 15.4, 15.5]
            },
            "metadata": {
                "operator": "김철수",
                "shift": "morning",
                "location": {
                    "building": "A",
                    "floor": 2,
                    "section": "paint-line-01"
                }
            },
            "alerts": [
                {"type": "warning", "message": "Temperature spike detected"},
                {"type": "info", "message": "Scheduled maintenance due"}
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "complex_logs.jsonl")
            config = {"logs": {"file_path": log_file_path}}
            
            save_issue_log(complex_log_data, config)
            
            # Verify complex data was saved correctly
            with open(log_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.loads(f.read().strip())
                assert saved_data == complex_log_data
                assert len(saved_data["measurements"]["temperature"]) == 4
                assert saved_data["metadata"]["operator"] == "김철수"
                assert len(saved_data["alerts"]) == 2

    def test_save_issue_log_empty_dict(self):
        """Test save_issue_log with empty dictionary"""
        empty_data = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "empty_logs.jsonl")
            config = {"logs": {"file_path": log_file_path}}
            
            save_issue_log(empty_data, config)
            
            # Verify empty data was saved correctly
            with open(log_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.loads(f.read().strip())
                assert saved_data == {}

    def test_save_issue_log_json_lines_format_verification(self):
        """Test that save_issue_log properly formats JSON Lines with newlines"""
        log_data_1 = {"machineId": "MACHINE_001", "issue": "Issue 1"}
        log_data_2 = {"machineId": "MACHINE_002", "issue": "Issue 2"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "jsonl_format_test.jsonl")
            config = {"logs": {"file_path": log_file_path}}
            
            save_issue_log(log_data_1, config)
            save_issue_log(log_data_2, config)
            
            # Read raw file content to verify formatting
            with open(log_file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Verify each line ends with newline
            lines = raw_content.split('\n')
            assert len(lines) == 3  # Two data lines + one empty line at end
            assert lines[2] == ""  # Last line should be empty due to trailing newline
            
            # Verify each line contains valid JSON
            json.loads(lines[0])  # Should not raise exception
            json.loads(lines[1])  # Should not raise exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])