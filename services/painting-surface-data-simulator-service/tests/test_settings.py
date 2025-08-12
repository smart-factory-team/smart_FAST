import pytest
import os
from app.config.settings import Settings


class TestSettings:
    """설정 클래스 테스트"""

    def test_settings_default_values(self):
        """기본값 테스트"""
        # 환경 변수가 설정되어 있으므로 기본값 확인 테스트
        settings = Settings()
        # 기본값 확인
        assert settings.azure_container_name == "simulator-data"
        assert settings.painting_data_folder == "painting-surface"
        assert settings.scheduler_interval_minutes == 1
        assert settings.batch_size == 10
        assert settings.painting_model_url == "http://painting-model-service:8002"
        assert settings.log_directory == "logs"
        assert settings.log_filename == "painting_defect_detections.json"
        assert settings.error_log_filename == "painting_errors.json"
        assert settings.http_timeout == 30
        assert settings.max_retries == 3

    def test_settings_with_environment_variables(self):
        """환경 변수를 통한 설정 테스트"""
        # 테스트 환경 변수 설정
        os.environ['AZURE_CONNECTION_STRING'] = 'test_connection_string'
        
        try:
            settings = Settings()
            
            # 기본값 확인
            assert settings.azure_container_name == "simulator-data"
            assert settings.painting_data_folder == "painting-surface"
            assert settings.scheduler_interval_minutes == 1
            assert settings.batch_size == 10
            assert settings.painting_model_url == "http://painting-model-service:8002"
            assert settings.log_directory == "logs"
            assert settings.log_filename == "painting_defect_detections.json"
            assert settings.error_log_filename == "painting_errors.json"
            assert settings.http_timeout == 30
            assert settings.max_retries == 3
            
            # 커스텀 값 확인
            assert settings.azure_connection_string == "test_connection_string"
            
        finally:
            # 환경 변수 정리
            if 'AZURE_CONNECTION_STRING' in os.environ:
                del os.environ['AZURE_CONNECTION_STRING']

    def test_model_service_url_property(self):
        """model_service_url 프로퍼티 테스트"""
        os.environ['AZURE_CONNECTION_STRING'] = 'test_connection_string'
        
        try:
            settings = Settings()
            assert settings.model_service_url == "http://painting-model-service:8002"
        finally:
            if 'AZURE_CONNECTION_STRING' in os.environ:
                del os.environ['AZURE_CONNECTION_STRING']

    def test_settings_custom_values(self):
        """커스텀 값 설정 테스트"""
        os.environ['AZURE_CONNECTION_STRING'] = 'test_connection_string'
        os.environ['AZURE_CONTAINER_NAME'] = 'custom-container'
        os.environ['PAINTING_DATA_FOLDER'] = 'custom-painting'
        os.environ['SCHEDULER_INTERVAL_MINUTES'] = '5'
        os.environ['BATCH_SIZE'] = '20'
        
        try:
            settings = Settings()
            
            assert settings.azure_container_name == "custom-container"
            assert settings.painting_data_folder == "custom-painting"
            assert settings.scheduler_interval_minutes == 5
            assert settings.batch_size == 20
            
        finally:
            # 환경 변수 정리
            for key in ['AZURE_CONNECTION_STRING', 'AZURE_CONTAINER_NAME', 
                       'PAINTING_DATA_FOLDER', 'SCHEDULER_INTERVAL_MINUTES', 'BATCH_SIZE']:
                if key in os.environ:
                    del os.environ[key]

    def test_settings_model_config(self):
        """모델 설정 테스트"""
        os.environ['AZURE_CONNECTION_STRING'] = 'test_connection_string'
        
        try:
            settings = Settings()
            
            assert settings.model_config["env_file"] == ".env"
            assert settings.model_config["env_file_encoding"] == "utf-8"
            
        finally:
            if 'AZURE_CONNECTION_STRING' in os.environ:
                del os.environ['AZURE_CONNECTION_STRING']
