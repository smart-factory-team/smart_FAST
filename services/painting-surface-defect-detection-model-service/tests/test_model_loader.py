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
        assert all(isinstance(k, int) for k in classes)
        
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
        # model_type이 있을 수 있으므로 제거하지 않음


    def test_load_yolo_model_invalid_file_extension(self):
        """잘못된 파일 확장자로 모델 로딩 실패 테스트"""
        with patch('app.models.yolo_model.hf_hub_download') as mock_hf_download:
            # 잘못된 확장자 파일 반환
            mock_hf_download.return_value = "/tmp/invalid_model.txt"
            
            with patch('app.models.yolo_model.YOLO') as mock_yolo:
                mock_yolo.side_effect = Exception("Invalid model file format")
                
                with pytest.raises(Exception, match="Invalid model file format"):
                    self.model_loader.load_yolo_model()

    def test_load_model_config_invalid_json(self):
        """잘못된 JSON 형식의 설정 파일 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_config_path = f.name
        
        try:
            with patch('app.models.yolo_model.hf_hub_download') as mock_hf_download:
                mock_hf_download.return_value = temp_config_path
                
                # JSON 파싱 오류로 인해 빈 딕셔너리 반환 예상
                result = self.model_loader.load_model_config()
                assert result == {}
        finally:
            os.unlink(temp_config_path)

    def test_load_model_config_empty_file(self):
        """빈 설정 파일 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")
            temp_config_path = f.name
        
        try:
            with patch('app.models.yolo_model.hf_hub_download') as mock_hf_download:
                mock_hf_download.return_value = temp_config_path
                
                result = self.model_loader.load_model_config()
                assert result == {}
        finally:
            os.unlink(temp_config_path)

    def test_load_class_mapping_invalid_json(self):
        """잘못된 JSON 형식의 클래스 매핑 파일 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json")
            temp_mapping_path = f.name
        
        try:
            with patch('app.models.yolo_model.hf_hub_download') as mock_hf_download:
                mock_hf_download.return_value = temp_mapping_path
                
                # JSON 파싱 오류로 인해 기본 매핑 반환 예상
                result = self.model_loader.load_class_mapping()
                assert result == self.model_loader.defect_classes
        finally:
            os.unlink(temp_mapping_path)

    def test_load_class_mapping_non_integer_keys(self):
        """정수가 아닌 키를 가진 클래스 매핑 테스트"""
        test_mapping = {
            "0": "dirt_class",
            "1": "runs_class", 
            "invalid_key": "invalid_value",
            "2.5": "float_key"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_mapping, f)
            temp_mapping_path = f.name
        
        try:
            with patch('app.models.yolo_model.hf_hub_download') as mock_hf_download:
                mock_hf_download.return_value = temp_mapping_path
                
                result = self.model_loader.load_class_mapping()
                
                # 정수로 변환 가능한 키만 포함되어야 함
                expected_result = {
                    0: "dirt_class",
                    1: "runs_class"
                }
                assert result == expected_result
        finally:
            os.unlink(temp_mapping_path)

    def test_load_threshold_config_invalid_json(self):
        """잘못된 JSON 형식의 임계값 설정 파일 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json")
            temp_threshold_path = f.name
        
        try:
            with patch('app.models.yolo_model.hf_hub_download') as mock_hf_download:
                mock_hf_download.return_value = temp_threshold_path
                
                result = self.model_loader.load_threshold_config()
                assert result == {}
        finally:
            os.unlink(temp_threshold_path)

    def test_load_threshold_config_with_extreme_values(self):
        """극한 값을 가진 임계값 설정 테스트"""
        test_thresholds = {
            "confidence_threshold": 0.0,    # 최소값
            "nms_threshold": 1.0,           # 최대값
            "iou_threshold": 0.999999       # 거의 1에 가까운 값
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_thresholds, f)
            temp_threshold_path = f.name
        
        try:
            with patch('app.models.yolo_model.hf_hub_download') as mock_hf_download:
                mock_hf_download.return_value = temp_threshold_path
                
                result = self.model_loader.load_threshold_config()
                assert result == test_thresholds
                assert result["confidence_threshold"] == 0.0
                assert result["nms_threshold"] == 1.0
                assert result["iou_threshold"] == 0.999999
        finally:
            os.unlink(temp_threshold_path)

    @patch('app.models.yolo_model.hf_hub_download')
    def test_model_config_persistence_across_calls(self, mock_hf_download):
        """여러 호출에서 모델 설정 지속성 테스트"""
        test_config = {
            "model_type": "yolo",
            "version": "2.0",
            "input_size": [1024, 1024],
            "batch_size": 32
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            mock_hf_download.return_value = temp_config_path
            
            # 첫 번째 로딩
            result1 = self.model_loader.load_model_config()
            assert result1 == test_config
            assert self.model_loader.model_config == test_config
            
            # 설정이 인스턴스에 저장되었는지 확인
            info = self.model_loader.get_model_info()
            assert info["config_loaded"] is True
            assert info["model_type"] == "yolo"
            assert info["version"] == "2.0"
            
        finally:
            os.unlink(temp_config_path)

    def test_environment_variable_combinations(self):
        """다양한 환경 변수 조합 테스트"""
        test_scenarios = [
            # 시나리오 1: ORG만 설정
            {
                'env_vars': {'HUGGING_FACE_ORG': 'custom_org'},
                'expected': {
                    'org': 'custom_org',
                    'repo': 'painting-surface-defect',
                    'model_name': 'custom_org/painting-surface-defect'
                }
            },
            # 시나리오 2: REPO만 설정  
            {
                'env_vars': {'HUGGING_FACE_REPO': 'custom_repo'},
                'expected': {
                    'org': '23smartfactory',
                    'repo': 'custom_repo', 
                    'model_name': '23smartfactory/custom_repo'
                }
            },
            # 시나리오 3: MODEL_NAME으로 오버라이드
            {
                'env_vars': {
                    'HUGGING_FACE_ORG': 'org1',
                    'HUGGING_FACE_REPO': 'repo1', 
                    'HUGGING_FACE_MODEL_NAME': 'completely/different'
                },
                'expected': {
                    'org': 'org1',
                    'repo': 'repo1',
                    'model_name': 'completely/different'
                }
            }
        ]
        
        for scenario in test_scenarios:
            with patch.dict(os.environ, scenario['env_vars'], clear=True):
                model_loader = PaintingSurfaceDefectModelLoader()
                
                assert model_loader.org == scenario['expected']['org']
                assert model_loader.repo == scenario['expected']['repo'] 
                assert model_loader.model_name == scenario['expected']['model_name']

    def test_defect_classes_data_integrity(self):
        """결함 클래스 데이터 무결성 테스트"""
        classes = self.model_loader.defect_classes
        
        # 모든 키가 정수인지 확인
        assert all(isinstance(k, int) for k in classes)
        
        # 모든 값이 비어있지 않은 문자열인지 확인
        assert all(isinstance(v, str) and len(v) > 0 for v in classes.values())
        
        # 키가 0부터 연속된 정수인지 확인
        sorted_keys = sorted(classes.keys())
        expected_keys = list(range(len(sorted_keys)))
        assert sorted_keys == expected_keys
        
        # 예상되는 클래스들이 모두 있는지 확인
        expected_classes = ["dirt", "runs", "scratch", "water_marks"]
        actual_values = list(classes.values())
        for expected_class in expected_classes:
            assert expected_class in actual_values

    @patch('app.models.yolo_model.hf_hub_download')
    @patch('app.models.yolo_model.YOLO')
    def test_model_loading_with_different_architectures(self, mock_yolo, mock_hf_download):
        """다양한 모델 아키텍처로 로딩 테스트"""
        mock_model_path = "/tmp/test_model.pt"
        mock_hf_download.return_value = mock_model_path
        
        # 다양한 YOLO 모델 타입 시뮬레이션
        model_types = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
        
        for model_type in model_types:
            mock_yolo_instance = MagicMock()
            mock_yolo_instance.model_type = model_type
            mock_yolo.return_value = mock_yolo_instance
            
            result = self.model_loader.load_yolo_model()
            assert result == mock_yolo_instance
            assert self.model_loader.model == mock_yolo_instance

    def test_get_model_info_data_types_and_structure(self):
        """모델 정보의 데이터 타입과 구조 테스트"""
        info = self.model_loader.get_model_info()
        
        # 필수 키들의 존재와 타입 확인
        type_checks = [
            ("model_name", str),
            ("org", str), 
            ("repo", str),
            ("defect_classes", dict),
            ("total_classes", int),
            ("model_loaded", bool),
            ("config_loaded", bool)
        ]
        
        for key, expected_type in type_checks:
            assert key in info, f"Key '{key}' missing from model info"
            assert isinstance(info[key], expected_type), f"Key '{key}' has wrong type, expected {expected_type}"
        
        # defect_classes의 내용 확인
        assert len(info["defect_classes"]) == 4
        assert info["total_classes"] == 4
        
        # 초기 상태에서는 모델과 설정이 로드되지 않았어야 함
        assert info["model_loaded"] is False
        assert info["config_loaded"] is False

    @patch('app.models.yolo_model.hf_hub_download')
    def test_error_handling_file_not_found(self, mock_hf_download):
        """파일을 찾을 수 없는 경우의 오류 처리 테스트"""
        mock_hf_download.side_effect = FileNotFoundError("File not found on Hugging Face Hub")
        
        with pytest.raises(FileNotFoundError, match="File not found on Hugging Face Hub"):
            self.model_loader.load_yolo_model()

    @patch('app.models.yolo_model.hf_hub_download')
    def test_error_handling_permission_denied(self, mock_hf_download):
        """권한 거부 오류 처리 테스트"""
        mock_hf_download.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError, match="Permission denied"):
            self.model_loader.load_yolo_model()

    @patch('app.models.yolo_model.hf_hub_download')  
    def test_error_handling_network_issues(self, mock_hf_download):
        """네트워크 관련 오류 처리 테스트"""
        import requests
        mock_hf_download.side_effect = requests.ConnectionError("Connection failed")
        
        with pytest.raises(requests.ConnectionError, match="Connection failed"):
            self.model_loader.load_yolo_model()

    def test_model_loader_string_representations(self):
        """모델 로더의 문자열 표현 테스트"""
        # __repr__ 메서드 테스트
        repr_str = repr(self.model_loader)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
        assert "PaintingSurfaceDefectModelLoader" in repr_str
        assert self.model_loader.model_name in repr_str

    def test_validate_model_with_mock_predict_failure(self):
        """predict 메서드 실패 시 모델 유효성 검사 테스트"""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Prediction failed")
        self.model_loader.model = mock_model
        
        result = self.model_loader.validate_model()
        assert result is False

    def test_validate_model_with_mock_predict_success(self):
        """predict 메서드 성공 시 모델 유효성 검사 테스트"""
        mock_model = MagicMock()
        mock_model.predict.return_value = MagicMock()  # 성공적인 예측 결과
        self.model_loader.model = mock_model
        
        result = self.model_loader.validate_model()
        assert result is True

    def test_model_config_with_nested_structures(self):
        """중첩된 구조를 가진 모델 설정 테스트"""
        complex_config = {
            "model_type": "yolo",
            "architecture": {
                "backbone": "CSPDarknet",
                "neck": "PANet",
                "head": "YOLOHead"
            },
            "training": {
                "optimizer": {
                    "type": "SGD",
                    "lr": 0.01,
                    "momentum": 0.9
                },
                "scheduler": {
                    "type": "StepLR",
                    "step_size": 30
                }
            },
            "hyperparameters": [
                {"name": "conf_thres", "value": 0.25},
                {"name": "iou_thres", "value": 0.45}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(complex_config, f)
            temp_config_path = f.name
        
        try:
            with patch('app.models.yolo_model.hf_hub_download') as mock_hf_download:
                mock_hf_download.return_value = temp_config_path
                
                result = self.model_loader.load_model_config()
                assert result == complex_config
                
                # 중첩된 구조 확인
                assert result["architecture"]["backbone"] == "CSPDarknet"
                assert result["training"]["optimizer"]["lr"] == 0.01
                assert len(result["hyperparameters"]) == 2
                
        finally:
            os.unlink(temp_config_path)

    def test_class_mapping_with_large_number_of_classes(self):
        """많은 수의 클래스를 가진 매핑 테스트"""
        large_mapping = {str(i): f"class_{i}" for i in range(100)}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(large_mapping, f)
            temp_mapping_path = f.name
        
        try:
            with patch('app.models.yolo_model.hf_hub_download') as mock_hf_download:
                mock_hf_download.return_value = temp_mapping_path
                
                result = self.model_loader.load_class_mapping()
                
                # 모든 클래스가 올바르게 변환되었는지 확인
                assert len(result) == 100
                for i in range(100):
                    assert i in result
                    assert result[i] == f"class_{i}"
                    
        finally:
            os.unlink(temp_mapping_path)

    def test_concurrent_method_calls(self):
        """여러 메서드의 동시 호출 테스트"""
        # 모델 정보를 여러 번 연속으로 호출
        info_results = []
        for _ in range(10):
            info = self.model_loader.get_model_info()
            info_results.append(info)
        
        # 모든 결과가 일관되어야 함
        first_result = info_results[0]
        for result in info_results[1:]:
            assert result == first_result

    def test_model_loader_attribute_isolation(self):
        """모델 로더 인스턴스 간 속성 격리 테스트"""
        # 두 번째 인스턴스 생성
        second_loader = PaintingSurfaceDefectModelLoader()
        
        # 첫 번째 인스턴스의 속성 수정
        test_model = MagicMock()
        test_config = {"test": "value"}
        
        self.model_loader.model = test_model
        self.model_loader.model_config = test_config
        
        # 두 번째 인스턴스가 영향받지 않았는지 확인
        assert second_loader.model is None
        assert second_loader.model_config is None
        assert second_loader.org == self.model_loader.org
        assert second_loader.repo == self.model_loader.repo

if __name__ == "__main__":
    pytest.main([__file__]) 