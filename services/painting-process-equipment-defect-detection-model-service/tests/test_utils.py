import json
import os
import tempfile
import pytest
from datetime import datetime
from unittest.mock import patch, mock_open
from pydantic import ValidationError
from app.services.utils import IssueLogInput, save_issue_log

# Test cases for IssueLogInput Pydantic model
class TestIssueLogInput:
    
    def test_valid_issue_log_input_creation(self):
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

    def test_invalid_issue_log_input_raises_error(self):
        invalid_data = {
            "machineId": 123,
            "timeStamp": "invalid-timestamp",
            "thick": "not_a_number",
            "voltage": 220.0,
            "current": 15.5,
            "temper": 65.2
        }
        with pytest.raises(ValidationError):
            IssueLogInput(**invalid_data)

# Test cases for save_issue_log function
class TestSaveIssueLog:
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_save_issue_log_success(self, mock_exists, mock_file):
        """정상적인 로그 저장이 잘 이루어지는지 테스트"""
        log_data = {"machineId": "MACHINE_001", "issue": "voltage_invalid"}
        config = {"logs": {"file_path": "/test/logs.jsonl"}}
        save_issue_log(log_data, config)
        
        mock_file.assert_called_with('/test/logs.jsonl', 'a', encoding='utf-8')

    @patch('builtins.print')
    def test_save_issue_log_no_file_path(self, mock_print):
        """설정 파일에 로그 경로가 없을 때 경고 메시지를 출력하는지 테스트"""
        log_data = {"machineId": "MACHINE_001"}
        config = {"logs": {}}
        save_issue_log(log_data, config)
        # `utils.py`의 실제 출력과 일치하도록 수정
        mock_print.assert_called_with("경고: 설정 파일에 로그 저장 경로(logs.file_path)가 지정되지 않았습니다.")

    @patch("os.makedirs")
    @patch('builtins.print')
    def test_save_issue_log_directory_creation_error(self, mock_print, mock_makedirs):
        """로그 디렉토리 생성 중 오류가 발생했을 때 경고를 출력하는지 테스트"""
        log_data = {"machineId": "MACHINE_001"}
        config = {"logs": {"file_path": "/test/non_existent_dir/test_logs.jsonl"}}
        
        mock_makedirs.side_effect = OSError("Permission denied")
        
        with patch("os.path.exists", return_value=False):
            save_issue_log(log_data, config)
                
        # `utils.py`의 실제 출력과 일치하도록 수정
        mock_print.assert_called_with(f"로그 파일 디렉토리 생성 중 오류 발생: {os.path.dirname(config['logs']['file_path'])}")

    @patch('builtins.open', new_callable=mock_open)
    @patch('builtins.print')
    def test_save_issue_log_file_write_error(self, mock_print, mock_file):
        """로그 파일에 쓰는 도중 오류가 발생했을 때 경고를 출력하는지 테스트"""
        log_data = {"machineId": "MACHINE_001"}
        config = {"logs": {"file_path": "/test/path/log.jsonl"}}
        mock_file.side_effect = OSError("Permission denied")
        
        save_issue_log(log_data, config)
        
        # `utils.py`의 실제 출력과 일치하도록 수정
        mock_print.assert_called_with(f"로그 파일 저장 중 오류 발생: {config['logs']['file_path']}")