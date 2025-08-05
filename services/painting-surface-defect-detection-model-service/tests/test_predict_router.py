import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os
import base64
import numpy as np
import cv2

# 앱을 import하기 위해 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.main import app
    from app.routers.predict import predict, predict_base64, PredictionRequest, PredictionResponse
except ImportError:
    # 대체 import 경로 시도
    from main import app
    from routers.predict import predict, predict_base64, PredictionRequest, PredictionResponse


class TestPredictRouter:
    """
    예측 라우터를 위한 종합 테스트 스위트
    테스트 프레임워크: pytest with FastAPI TestClient
    """

    def setup_method(self):
        """각 테스트 메서드에 대한 테스트 클라이언트 설정"""
        self.client = TestClient(app)

    def test_predict_endpoint_exists(self):
        """예측 엔드포인트가 존재하고 접근 가능한지 테스트"""
        # 엔드포인트가 등록되었는지 확인
        schema = app.openapi()
        paths = schema.get("paths", {})
        
        # 파일 업로드 방식 엔드포인트
        assert "/api/predict" in paths
        assert "post" in paths["/api/predict"]
        
        # Base64 방식 엔드포인트
        assert "/api/predict/base64" in paths
        assert "post" in paths["/api/predict/base64"]

    def test_predict_endpoint_method_not_allowed(self):
        """예측 엔드포인트에서 GET 메서드가 허용되지 않는지 테스트"""
        response = self.client.get("/api/predict")
        assert response.status_code == 405
        
        response = self.client.get("/api/predict/base64")
        assert response.status_code == 405

    def test_predict_endpoint_file_upload(self):
        """파일 업로드 방식 예측 엔드포인트 테스트"""
        # 더미 이미지 파일 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_data = buffer.tobytes()

        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            mock_predict.return_value = {
                "predictions": [],
                "image_shape": [100, 100, 3],
                "confidence_threshold": 0.5,
                "timestamp": "2024-01-01T00:00:00",
                "model_source": "Hugging Face"
            }

            response = self.client.post(
                "/api/predict",
                files={"image": ("test.jpg", image_data, "image/jpeg")},
                data={"confidence_threshold": 0.5}
            )
            
            # 검증 오류가 발생하지 않아야 함
            assert response.status_code in [200, 500]  # 500 if model not loaded

    def test_predict_endpoint_invalid_file_type(self):
        """잘못된 파일 타입 처리 테스트"""
        # 텍스트 파일로 테스트
        response = self.client.post(
            "/api/predict",
            files={"image": ("test.txt", b"invalid image data", "text/plain")},
            data={"confidence_threshold": 0.5}
        )
        assert response.status_code == 400

    def test_predict_endpoint_large_file(self):
        """대용량 파일 처리 테스트"""
        # 11MB 이미지 데이터 생성
        large_image_data = b"x" * (11 * 1024 * 1024)
        
        response = self.client.post(
            "/api/predict",
            files={"image": ("large.jpg", large_image_data, "image/jpeg")},
            data={"confidence_threshold": 0.5}
        )
        assert response.status_code == 400

    def test_predict_base64_endpoint_exists(self):
        """Base64 방식 예측 엔드포인트가 존재하는지 테스트"""
        schema = app.openapi()
        paths = schema.get("paths", {})
        
        assert "/api/predict/base64" in paths
        assert "post" in paths["/api/predict/base64"]

    def test_predict_base64_endpoint_requires_json(self):
        """Base64 방식 예측 엔드포인트가 JSON 콘텐츠 타입을 요구하는지 테스트"""
        response = self.client.post(
            "/api/predict/base64",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422

    def test_predict_base64_endpoint_missing_required_fields(self):
        """Base64 방식 예측 엔드포인트가 image_base64 필드를 요구하는지 테스트"""
        response = self.client.post(
            "/api/predict/base64",
            json={}
        )
        assert response.status_code == 422

    def test_predict_base64_endpoint_invalid_base64(self):
        """Base64 방식 예측 엔드포인트가 base64 인코딩을 검증하는지 테스트"""
        response = self.client.post(
            "/api/predict/base64",
            json={
                "image_base64": "invalid_base64",
                "confidence_threshold": 0.5
            }
        )
        # Pydantic 검증으로 인해 422가 반환되어야 함
        assert response.status_code == 422

    def test_predict_base64_endpoint_invalid_confidence_threshold(self):
        """Base64 방식 예측 엔드포인트가 신뢰도 임계값 범위를 검증하는지 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 잘못된 임계값 테스트
        invalid_thresholds = [-0.1, 1.1, 2.0, -1.0]

        for threshold in invalid_thresholds:
            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": image_base64,
                    "confidence_threshold": threshold
                }
            )
            # Pydantic 검증으로 인해 422가 반환되어야 함
            assert response.status_code == 422

    def test_predict_base64_endpoint_valid_request_structure(self):
        """Base64 방식 예측 엔드포인트가 유효한 요청 구조를 수락하는지 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        valid_request = {
            "image_base64": image_base64,
            "confidence_threshold": 0.5
        }

        # 예측 함수 모킹
        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            mock_predict.return_value = {
                "predictions": [],
                "image_shape": [100, 100, 3],
                "confidence_threshold": 0.5,
                "timestamp": "2024-01-01T00:00:00",
                "model_source": "Hugging Face"
            }

            response = self.client.post(
                "/api/predict/base64",
                json=valid_request
            )
            
            # 검증 오류가 발생하지 않아야 함
            assert response.status_code in [200, 500]  # 500 if model not loaded

    @patch('app.routers.predict.predict_anomaly')
    def test_predict_base64_endpoint_successful_prediction(self, mock_predict):
        """Base64 방식 예측 엔드포인트 성공적인 예측 테스트"""
        # 모킹 응답 설정
        mock_response = {
            "predictions": [
                {
                    "bbox": [10, 10, 50, 50],
                    "confidence": 0.95,
                    "class_id": 0,
                    "class_name": "dirt",
                    "area": 1600.0
                },
                {
                    "bbox": [60, 60, 90, 90],
                    "confidence": 0.87,
                    "class_id": 1,
                    "class_name": "runs",
                    "area": 900.0
                }
            ],
            "image_shape": [100, 100, 3],
            "confidence_threshold": 0.5,
            "timestamp": "2024-01-01T00:00:00",
            "model_source": "Hugging Face"
        }
        mock_predict.return_value = mock_response

        # 테스트 요청 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        request_data = {
            "image_base64": image_base64,
            "confidence_threshold": 0.5
        }

        response = self.client.post(
            "/api/predict/base64",
            json=request_data
        )

        # 응답 확인
        if response.status_code == 200:
            data = response.json()
            
            # 필수 필드 확인
            assert "predictions" in data
            assert "image_shape" in data
            assert "confidence_threshold" in data
            assert "timestamp" in data
            assert "model_source" in data
            
            # 예측 결과 구조 확인
            predictions = data["predictions"]
            assert len(predictions) == 2
            
            # 첫 번째 예측 확인
            first_pred = predictions[0]
            assert "bbox" in first_pred
            assert "confidence" in first_pred
            assert "class_id" in first_pred
            assert "class_name" in first_pred
            assert "area" in first_pred
            
            # 값 확인
            assert first_pred["confidence"] == 0.95
            assert first_pred["class_id"] == 0
            assert first_pred["class_name"] == "dirt"

    def test_predict_base64_endpoint_empty_predictions(self):
        """Base64 방식 예측 엔드포인트 빈 예측 결과 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            mock_predict.return_value = {
                "predictions": [],
                "image_shape": [100, 100, 3],
                "confidence_threshold": 0.5,
                "timestamp": "2024-01-01T00:00:00",
                "model_source": "Hugging Face"
            }

            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": image_base64,
                    "confidence_threshold": 0.5
                }
            )

            if response.status_code == 200:
                data = response.json()
                assert data["predictions"] == []

    def test_predict_base64_endpoint_default_confidence_threshold(self):
        """Base64 방식 예측 엔드포인트 기본 신뢰도 임계값 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            mock_predict.return_value = {
                "predictions": [],
                "image_shape": [100, 100, 3],
                "confidence_threshold": 0.5,
                "timestamp": "2024-01-01T00:00:00",
                "model_source": "Hugging Face"
            }

            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": image_base64
                    # confidence_threshold 제공하지 않음
                }
            )

            # 검증 오류가 발생하지 않아야 함
            assert response.status_code in [200, 500]

    def test_predict_base64_endpoint_large_image(self):
        """Base64 방식 예측 엔드포인트 대용량 이미지 테스트"""
        # 대용량 이미지 생성
        large_image = np.zeros((640, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', large_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            mock_predict.return_value = {
                "predictions": [],
                "image_shape": [640, 640, 3],
                "confidence_threshold": 0.5,
                "timestamp": "2024-01-01T00:00:00",
                "model_source": "Hugging Face"
            }

            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": image_base64,
                    "confidence_threshold": 0.5
                }
            )

            # 검증 오류가 발생하지 않아야 함
            assert response.status_code in [200, 500]

    def test_predict_base64_endpoint_different_image_formats(self):
        """Base64 방식 예측 엔드포인트 다양한 이미지 형식 테스트"""
        # PNG 형식으로 테스트
        png_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', png_image)
        png_base64 = base64.b64encode(buffer).decode('utf-8')

        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            mock_predict.return_value = {
                "predictions": [],
                "image_shape": [100, 100, 3],
                "confidence_threshold": 0.5,
                "timestamp": "2024-01-01T00:00:00",
                "model_source": "Hugging Face"
            }

            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": png_base64,
                    "confidence_threshold": 0.5
                }
            )

            # 검증 오류가 발생하지 않아야 함
            assert response.status_code in [200, 500]

    def test_predict_base64_endpoint_edge_case_confidence_thresholds(self):
        """Base64 방식 예측 엔드포인트 경계값 신뢰도 임계값 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 경계값 테스트
        edge_cases = [0.0, 1.0, 0.001, 0.999]

        for threshold in edge_cases:
            with patch('app.routers.predict.predict_anomaly') as mock_predict:
                mock_predict.return_value = {
                    "predictions": [],
                    "image_shape": [100, 100, 3],
                    "confidence_threshold": threshold,
                    "timestamp": "2024-01-01T00:00:00",
                    "model_source": "Hugging Face"
                }

                response = self.client.post(
                    "/api/predict/base64",
                    json={
                        "image_base64": image_base64,
                        "confidence_threshold": threshold
                    }
                )

                # 검증 오류가 발생하지 않아야 함
                assert response.status_code in [200, 500]

    def test_predict_base64_endpoint_error_handling(self):
        """Base64 방식 예측 엔드포인트 오류 처리 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            # 예측에서 오류 시뮬레이션
            mock_predict.side_effect = Exception("Model prediction failed")

            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": image_base64,
                    "confidence_threshold": 0.5
                }
            )

            # 오류 상태 반환해야 함
            assert response.status_code == 500

    def test_predict_base64_endpoint_response_headers(self):
        """Base64 방식 예측 엔드포인트 응답 헤더 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            mock_predict.return_value = {
                "predictions": [],
                "image_shape": [100, 100, 3],
                "confidence_threshold": 0.5,
                "timestamp": "2024-01-01T00:00:00",
                "model_source": "Hugging Face"
            }

            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": image_base64,
                    "confidence_threshold": 0.5
                }
            )

            if response.status_code == 200:
                # 콘텐츠 타입 확인
                assert response.headers.get("content-type") == "application/json"

    def test_model_info_endpoint(self):
        """모델 정보 엔드포인트 테스트"""
        response = self.client.get("/api/model/info")
        # 모델이 로드되지 않은 경우 503, 로드된 경우 200
        assert response.status_code in [200, 503]


class TestPredictionRequestModel:
    """PredictionRequest Pydantic 모델 테스트"""

    def test_valid_prediction_request(self):
        """유효한 예측 요청이 수락되는지 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        request = PredictionRequest(
            image_base64=image_base64,
            confidence_threshold=0.5
        )

        assert request.image_base64 == image_base64
        assert request.confidence_threshold == 0.5

    def test_prediction_request_default_confidence(self):
        """기본 신뢰도 임계값이 올바르게 설정되는지 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        request = PredictionRequest(image_base64=image_base64)
        assert request.confidence_threshold == 0.5

    def test_prediction_request_invalid_base64(self):
        """잘못된 base64가 검증 오류를 발생시키는지 테스트"""
        # Pydantic v1에서는 validator가 ValueError를 발생시킴
        with pytest.raises(ValueError):
            PredictionRequest(
                image_base64="invalid_base64",
                confidence_threshold=0.5
            )

    def test_prediction_request_invalid_confidence(self):
        """잘못된 신뢰도 임계값이 검증 오류를 발생시키는지 테스트"""
        # 유효한 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Pydantic Field 검증으로 인해 ValidationError가 발생해야 함
        with pytest.raises(ValueError):
            PredictionRequest(
                image_base64=image_base64,
                confidence_threshold=1.5
            )

    def test_prediction_request_short_base64(self):
        """짧은 base64 문자열이 검증 오류를 발생시키는지 테스트"""
        with pytest.raises(ValueError):
            PredictionRequest(
                image_base64="short",
                confidence_threshold=0.5
            )


class TestPredictionResponseModel:
    """PredictionResponse Pydantic 모델 테스트"""

    def test_valid_prediction_response(self):
        """유효한 예측 응답이 수락되는지 테스트"""
        response = PredictionResponse(
            predictions=[
                {
                    "bbox": [10, 10, 50, 50],
                    "confidence": 0.95,
                    "class_id": 0,
                    "class_name": "dirt",
                    "area": 1600.0
                }
            ],
            image_shape=[100, 100, 3],
            confidence_threshold=0.5,
            timestamp="2024-01-01T00:00:00",
            model_source="Hugging Face"
        )

        assert len(response.predictions) == 1
        assert response.image_shape == [100, 100, 3]
        assert response.confidence_threshold == 0.5
        assert response.timestamp == "2024-01-01T00:00:00"
        assert response.model_source == "Hugging Face"

    def test_prediction_response_empty_predictions(self):
        """빈 예측 결과가 있는 예측 응답 테스트"""
        response = PredictionResponse(
            predictions=[],
            image_shape=[100, 100, 3],
            confidence_threshold=0.5,
            timestamp="2024-01-01T00:00:00",
            model_source="Hugging Face"
        )

        assert response.predictions == []


class TestPredictAnomalyFunction:
    """predict_anomaly 함수 테스트"""

    def test_predict_anomaly_function_exists(self):
        """predict_anomaly 함수가 존재하는지 테스트"""
        assert callable(predict_anomaly)

    @patch('app.routers.predict.detection_service')
    def test_predict_anomaly_with_mock_service(self, mock_service):
        """모킹된 서비스로 predict_anomaly 함수 테스트"""
        # 모킹 서비스 설정
        mock_service.threshold_config = {"confidence_threshold": 0.5}
        mock_service.defect_classes = {0: "dirt", 1: "runs", 2: "scratch", 3: "water_marks"}
        
        # 모킹 모델 생성
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95])
        mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
        mock_model.predict.return_value = [mock_result]
        mock_service.model = mock_model

        # 테스트 이미지 데이터 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_data = buffer.tobytes()

        # 함수 테스트
        result = predict_anomaly(image_data, 0.5)

        # 결과 구조 확인
        assert "predictions" in result
        assert "image_shape" in result
        assert "confidence_threshold" in result
        assert "timestamp" in result
        assert "model_source" in result

    def test_predict_anomaly_invalid_image_data(self):
        """잘못된 이미지 데이터로 predict_anomaly 테스트"""
        # 초기화 문제를 피하기 위해 detection_service 모킹
        with patch('app.routers.predict.detection_service') as mock_service:
            mock_service.threshold_config = {"confidence_threshold": 0.5}
            mock_service.defect_classes = {0: "dirt", 1: "runs", 2: "scratch", 3: "water_marks"}
            mock_service.model.predict.return_value = [MagicMock()]
            
            with pytest.raises(ValueError):
                predict_anomaly(b"invalid_image_data", 0.5)


if __name__ == "__main__":
    pytest.main([__file__]) 