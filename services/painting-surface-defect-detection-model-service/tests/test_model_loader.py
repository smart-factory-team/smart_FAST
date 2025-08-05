import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
import json

# 앱을 import하기 위해 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.models.yolo_model import PaintingSurfaceDefectModelLoader
except ImportError:
    # 대체 import 경로 시도
    from models.yolo_model import PaintingSurfaceDefectModelLoader


class TestPaintingSurfaceDefectModelLoader:
    """
    PaintingSurfaceDefectModelLoader 클래스를 위한 테스트 스위트
    """

    def setup_method(self):
        """각 테스트 메서드에 대한 테스트 환경 설정"""
        self.model_loader = PaintingSurfaceDefectModelLoader()

    def test_model_loader_initialization(self):
        """모델 로더가 올바르게 초기화되는지 테스트"""
        assert self.model_loader.org == "23smartfactory"
        assert self.model_loader.repo == "painting-surface-defect"
        assert self.model_loader.model_name == "23smartfactory/painting-surface-defect"
        assert self.model_loader.model is None
        assert self.model_loader.model_config is None

    def test_default_defect_classes(self):
        """기본 결함 클래스가 올바르게 설정되는지 테스트"""
        expected_classes = {
            0: "dirt",           # 먼지/오염
            1: "runs",           # 흘러내림
            2: "scratch",        # 스크래치
            3: "water_marks"     # 물 자국
        }
        
        assert self.model_loader.defect_classes == expected_classes

    @patch('app.models.yolo_model.hf_hub_download')
    @patch('app.models.yolo_model.YOLO')
    def test_load_yolo_model_success(self, mock_yolo, mock_hf_download):
        """YOLO 모델 로딩 성공 테스트"""
        # 모킹 설정
        mock_model_path = "/tmp/test_model.pt"
        mock_hf_download.return_value = mock_model_path
        
        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance

        # 모델 로딩 테스트
        result = self.model_loader.load_yolo_model()

        # 모킹이 올바르게 호출되었는지 확인
        mock_hf_download.assert_called_once_with(
            repo_id="23smartfactory/painting-surface-defect",
            filename="paint_defect_yolov8m_bbox_best.pt"
        )
        mock_yolo.assert_called_once_with(mock_model_path)

        # 결과 확인
        assert result == mock_yolo_instance
        assert self.model_loader.model == mock_yolo_instance

    @patch('app.models.yolo_model.hf_hub_download')
    def test_load_yolo_model_download_failure(self, mock_hf_download):
        """YOLO 모델 다운로드 실패 시 테스트"""
        # 모킹에서 예외 발생 설정
        mock_hf_download.side_effect = Exception("Download failed")

        # 예외가 발생하는지 테스트
        with pytest.raises(Exception, match="Download failed"):
            self.model_loader.load_yolo_model()

    @patch('app.models.yolo_model.hf_hub_download')
    def test_load_model_config_success(self, mock_hf_download):
        """모델 설정 파일 로딩 성공 테스트"""
        # 모킹 설정
        mock_config_path = "/tmp/model_config.json"
        mock_hf_download.return_value = mock_config_path
        
        # 임시 설정 파일 생성
        test_config = {
            "model_type": "yolo",
            "version": "1.0",
            "input_size": [640, 640],
            "num_classes": 4
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_config_path = f.name

        try:
            mock_hf_download.return_value = temp_config_path
            
            # 설정 로딩 테스트
            result = self.model_loader.load_model_config()

            # 모킹이 올바르게 호출되었는지 확인
            mock_hf_download.assert_called_once_with(
                repo_id="23smartfactory/painting-surface-defect",
                filename="model_config.json"
            )

            # 결과 확인
            assert result == test_config

        finally:
            # 정리
            os.unlink(temp_config_path)

    @patch('app.models.yolo_model.hf_hub_download')
    def test_load_model_config_not_found(self, mock_hf_download):
        """모델 설정 파일을 찾을 수 없을 때 테스트"""
        # 모킹에서 예외 발생 설정
        mock_hf_download.side_effect = Exception("File not found")

        # 기본 설정이 반환되는지 테스트
        result = self.model_loader.load_model_config()

        # 빈 딕셔너리가 반환되어야 함
        assert result == {}

    @patch('app.models.yolo_model.hf_hub_download')
    def test_load_class_mapping_success(self, mock_hf_download):
        """클래스 매핑 파일 로딩 성공 테스트"""
        # 모킹 설정
        mock_mapping_path = "/tmp/class_mapping.json"
        mock_hf_download.return_value = mock_mapping_path
        
        # 임시 매핑 파일 생성
        test_mapping = {
            "0": "dirt",
            "1": "runs",
            "2": "scratch",
            "3": "water_marks"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_mapping, f)
            temp_mapping_path = f.name

        try:
            mock_hf_download.return_value = temp_mapping_path
            
            # 매핑 로딩 테스트
            result = self.model_loader.load_class_mapping()

            # 모킹이 올바르게 호출되었는지 확인
            mock_hf_download.assert_called_once_with(
                repo_id="23smartfactory/painting-surface-defect",
                filename="class_mapping.json"
            )

            # 결과 확인
            expected_mapping = {int(k): v for k, v in test_mapping.items()}
            assert result == expected_mapping

        finally:
            # 정리
            os.unlink(temp_mapping_path)

    @patch('app.models.yolo_model.hf_hub_download')
    def test_load_class_mapping_not_found(self, mock_hf_download):
        """클래스 매핑 파일을 찾을 수 없을 때 테스트"""
        # 모킹에서 예외 발생 설정
        mock_hf_download.side_effect = Exception("File not found")

        # 기본 매핑이 반환되는지 테스트
        result = self.model_loader.load_class_mapping()

        # 기본 매핑이 반환되어야 함
        assert result == self.model_loader.defect_classes

    @patch('app.models.yolo_model.hf_hub_download')
    def test_load_threshold_config_success(self, mock_hf_download):
        """임계값 설정 파일 로딩 성공 테스트"""
        # 모킹 설정
        mock_threshold_path = "/tmp/thresholds.json"
        mock_hf_download.return_value = mock_threshold_path
        
        # 임시 임계값 파일 생성
        test_thresholds = {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "iou_threshold": 0.5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_thresholds, f)
            temp_threshold_path = f.name

        try:
            mock_hf_download.return_value = temp_threshold_path
            
            # 임계값 로딩 테스트
            result = self.model_loader.load_threshold_config()

            # 모킹이 올바르게 호출되었는지 확인
            mock_hf_download.assert_called_once_with(
                repo_id="23smartfactory/painting-surface-defect",
                filename="thresholds.json"
            )

            # 결과 확인
            assert result == test_thresholds

        finally:
            # 정리
            os.unlink(temp_threshold_path)

    @patch('app.models.yolo_model.hf_hub_download')
    def test_load_threshold_config_not_found(self, mock_hf_download):
        """임계값 설정 파일을 찾을 수 없을 때 테스트"""
        # 모킹에서 예외 발생 설정
        mock_hf_download.side_effect = Exception("File not found")

        # 빈 딕셔너리가 반환되는지 테스트
        result = self.model_loader.load_threshold_config()

        # 빈 딕셔너리가 반환되어야 함
        assert result == {}

    def test_get_model_info(self):
        """모델 정보가 올바르게 반환되는지 테스트"""
        info = self.model_loader.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "model_type" in info
        assert "org" in info
        assert "repo" in info
        assert "defect_classes" in info
        assert "total_classes" in info
        assert "model_loaded" in info
        assert "config_loaded" in info

    @patch('app.models.yolo_model.hf_hub_download')
    @patch('app.models.yolo_model.YOLO')
    def test_validate_model_success(self, mock_yolo, mock_hf_download):
        """모델 유효성 검사 성공 테스트"""
        # 모킹 설정
        mock_model_path = "/tmp/test_model.pt"
        mock_hf_download.return_value = mock_model_path
        
        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance
        self.model_loader.model = mock_yolo_instance

        # 유효성 검사 테스트
        result = self.model_loader.validate_model()

        # 결과 확인
        assert result is True

    def test_validate_model_no_model(self):
        """모델이 로드되지 않았을 때 유효성 검사 테스트"""
        self.model_loader.model = None

        # 유효성 검사 테스트
        result = self.model_loader.validate_model()

        # 결과 확인
        assert result is False

    @patch('app.models.yolo_model.hf_hub_download')
    @patch('app.models.yolo_model.YOLO')
    def test_validate_model_invalid_model(self, mock_yolo, mock_hf_download):
        """잘못된 모델로 유효성 검사 테스트"""
        # 모킹 설정
        mock_model_path = "/tmp/test_model.pt"
        mock_hf_download.return_value = mock_model_path
        
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.predict.side_effect = Exception("Model error")
        mock_yolo.return_value = mock_yolo_instance
        self.model_loader.model = mock_yolo_instance

        # 유효성 검사 테스트
        result = self.model_loader.validate_model()

        # 결과 확인
        assert result is False

    def test_environment_variable_override(self):
        """환경 변수로 기본값을 오버라이드할 수 있는지 테스트"""
        # 환경 변수로 테스트
        with patch.dict(os.environ, {
            'HUGGING_FACE_ORG': 'test_org',
            'HUGGING_FACE_REPO': 'test_repo',
            'HUGGING_FACE_MODEL_NAME': 'test_org/test_repo'
        }):
            model_loader = PaintingSurfaceDefectModelLoader()
            
            assert model_loader.org == "test_org"
            assert model_loader.repo == "test_repo"
            assert model_loader.model_name == "test_org/test_repo"

    def test_model_loader_attributes(self):
        """필요한 모든 속성이 존재하는지 테스트"""
        required_attributes = [
            'org', 'repo', 'model_name', 'defect_classes', 
            'model', 'model_config'
        ]
        
        for attr in required_attributes:
            assert hasattr(self.model_loader, attr)

    def test_defect_classes_structure(self):
        """결함 클래스가 올바른 구조를 가지고 있는지 테스트"""
        classes = self.model_loader.defect_classes
        
        # 모든 키가 정수인지 확인
        assert all(isinstance(k, int) for k in classes.keys())
        
        # 모든 값이 문자열인지 확인
        assert all(isinstance(v, str) for v in classes.values())
        
        # 예상되는 클래스 수 확인
        assert len(classes) == 4

    def test_model_loader_repr(self):
        """모델 로더가 의미있는 문자열 표현을 가지고 있는지 테스트"""
        repr_str = repr(self.model_loader)
        
        assert "PaintingSurfaceDefectModelLoader" in repr_str
        assert "23smartfactory" in repr_str
        assert "painting-surface-defect" in repr_str

    def test_model_info_with_config(self):
        """설정이 로드된 상태에서 모델 정보 테스트"""
        # 설정 설정
        self.model_loader.model_config = {
            "model_type": "yolo",
            "version": "1.0",
            "input_size": [640, 640],
            "num_classes": 4
        }
        
        info = self.model_loader.get_model_info()
        
        # 설정 정보가 포함되어 있는지 확인
        assert info["model_type"] == "yolo"
        assert info["version"] == "1.0"
        assert info["input_size"] == [640, 640]
        assert info["num_classes"] == 4
        assert info["config_loaded"] is True

    def test_model_info_without_config(self):
        """설정이 로드되지 않은 상태에서 모델 정보 테스트"""
        # 설정이 없는 상태
        self.model_loader.model_config = None
        
        info = self.model_loader.get_model_info()
        
        # 기본 정보만 포함되어 있는지 확인
        assert info["config_loaded"] is False
        assert "model_type" not in info
        assert "version" not in info


if __name__ == "__main__":
    pytest.main([__file__]) 