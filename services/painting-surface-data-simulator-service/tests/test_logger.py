import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from app.utils.logger import AnomalyLogger


class TestAnomalyLogger:
    """AnomalyLogger 클래스 테스트"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        self.test_log_dir = "test-logs"
        self.test_log_file = os.path.join(self.test_log_dir, "test_log.json")
        self.test_error_file = os.path.join(self.test_log_dir, "test_error.json")

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리"""
        # 테스트 로그 파일 정리
        if os.path.exists(self.test_log_dir):
            import shutil
            shutil.rmtree(self.test_log_dir)

    @patch('app.utils.logger.settings')
    def test_logger_initialization(self, mock_settings):
        """로거 초기화 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "test_log.json"
        
        logger = AnomalyLogger()
        
        assert logger.log_file_path == self.test_log_file
        assert os.path.exists(self.test_log_dir)

    @patch('app.utils.logger.settings')
    def test_log_anomaly(self, mock_settings):
        """결함 탐지 로그 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "test_log.json"
        
        logger = AnomalyLogger()
        
        # 테스트 데이터
        service_name = "test-service"
        prediction_result = {
            "defect_count": 2,
            "total_count": 5,
            "status": "anomaly"
        }
        original_data = {"image_path": "test.jpg"}
        
        # 로그 기록
        logger.log_anomaly(service_name, prediction_result, original_data)
        
        # 로그 파일 확인
        assert os.path.exists(self.test_log_file)
        
        with open(self.test_log_file, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline().strip())
            
        assert log_entry["service_name"] == service_name
        assert log_entry["prediction"] == prediction_result
        assert log_entry["original_data"] == original_data
        assert "timestamp" in log_entry

    @patch('app.utils.logger.settings')
    def test_log_normal_processing(self, mock_settings):
        """정상 처리 로그 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "test_log.json"
        
        logger = AnomalyLogger()
        
        # 테스트 데이터
        service_name = "test-service"
        prediction_result = {
            "defect_count": 0,
            "total_count": 5,
            "status": "normal"
        }
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            logger.log_normal_processing(service_name, prediction_result)
            
            # 콘솔 출력 확인
            mock_print.assert_called()

    @patch('app.utils.logger.settings')
    def test_log_error(self, mock_settings):
        """에러 로그 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.error_log_filename = "test_error.json"
        
        logger = AnomalyLogger()
        
        # 테스트 데이터
        service_name = "test-service"
        error_message = "Test error message"
        original_data = {"image_path": "test.jpg"}
        
        # 에러 로그 기록
        logger.log_error(service_name, error_message, original_data)
        
        # 에러 로그 파일 확인
        assert os.path.exists(self.test_error_file)
        
        with open(self.test_error_file, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline().strip())
            
        assert log_entry["service_name"] == service_name
        assert log_entry["error"] == error_message
        assert log_entry["original_data"] == original_data
        assert "timestamp" in log_entry

    @patch('app.utils.logger.settings')
    def test_log_error_without_original_data(self, mock_settings):
        """원본 데이터 없이 에러 로그 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.error_log_filename = "test_error.json"
        
        logger = AnomalyLogger()
        
        # 테스트 데이터
        service_name = "test-service"
        error_message = "Test error message"
        
        # 에러 로그 기록 (원본 데이터 없음)
        logger.log_error(service_name, error_message)
        
        # 에러 로그 파일 확인
        assert os.path.exists(self.test_error_file)
        
        with open(self.test_error_file, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline().strip())
            
        assert log_entry["service_name"] == service_name
        assert log_entry["error"] == error_message
        assert log_entry["original_data"] is None

    @patch('app.utils.logger.settings')
    def test_multiple_log_entries(self, mock_settings):
        """여러 로그 항목 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "test_log.json"
        
        logger = AnomalyLogger()
        
        # 여러 로그 기록
        for i in range(3):
            service_name = f"service-{i}"
            prediction_result = {
                "defect_count": i,
                "total_count": 5,
                "status": "anomaly" if i > 0 else "normal"
            }
            original_data = {"image_path": f"test_{i}.jpg"}
            
            logger.log_anomaly(service_name, prediction_result, original_data)
        
        # 로그 파일에서 모든 항목 확인
        with open(self.test_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        assert len(lines) == 3
        
        for i, line in enumerate(lines):
            log_entry = json.loads(line.strip())
            assert log_entry["service_name"] == f"service-{i}"
            assert log_entry["prediction"]["defect_count"] == i

    @patch('app.utils.logger.settings')
    def test_logger_with_existing_directory(self, mock_settings):
        """기존 디렉토리가 있는 경우 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "test_log.json"
        
        # 디렉토리 미리 생성
        os.makedirs(self.test_log_dir, exist_ok=True)
        
        logger = AnomalyLogger()
        
        # 로거가 정상적으로 작동하는지 확인
        assert logger.log_file_path == self.test_log_file
        
        # 로그 기록 테스트
        service_name = "test-service"
        prediction_result = {"defect_count": 1, "total_count": 5, "status": "anomaly"}
        original_data = {"image_path": "test.jpg"}
        
        logger.log_anomaly(service_name, prediction_result, original_data)
        
        assert os.path.exists(self.test_log_file)
