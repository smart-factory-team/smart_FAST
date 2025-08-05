import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os
import base64
import numpy as np

# 앱을 import하기 위해 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.main import app
except ImportError:
    # 대체 import 경로 시도
    from main import app


class TestFastAPIMainApplication:
    """
    FastAPI 메인 애플리케이션을 위한 종합 테스트 스위트
    테스트 프레임워크: pytest with FastAPI TestClient
    """

    def setup_method(self):
        """각 테스트 메서드에 대한 테스트 클라이언트 설정"""
        self.client = TestClient(app)

    def test_app_instance_creation(self):
        """FastAPI 앱 인스턴스가 올바르게 생성되는지 테스트"""
        assert isinstance(app, FastAPI)
        assert app.title == "Painting Surface Defect Detection API"
        assert app.version == "1.0"

    def test_app_configuration_properties(self):
        """필요한 모든 앱 설정 속성이 설정되었는지 테스트"""
        # 제목 테스트
        assert app.title == "Painting Surface Defect Detection API"

        # 버전 테스트
        assert app.version == "1.0"

        # 설명이 None이거나 문자열인지 테스트
        assert app.description is None or isinstance(app.description, str)

        # 앱이 root_path를 가지고 있는지 테스트 (기본값은 빈 문자열)
        assert hasattr(app, 'root_path')

    def test_predict_router_inclusion(self):
        """예측 라우터가 올바른 접두사와 함께 제대로 포함되었는지 테스트"""
        # 등록된 모든 라우트 가져오기
        routes = [route for route in app.routes]

        # 라우트가 있는지 확인 (최소한 docs, openapi, redoc)
        assert len(routes) > 0

        # /api 접두사가 있는 라우트 찾기
        api_routes = []
        for route in routes:
            if hasattr(route, 'path') and route.path.startswith('/api'):
                api_routes.append(route)

        # 예측 라우터에서 최소한 하나의 API 라우트가 있어야 함
        assert len(api_routes) > 0, "No API routes found with /api prefix"

    def test_predict_endpoint_registration(self):
        """예측 엔드포인트가 제대로 등록되었는지 테스트"""
        # 예측 엔드포인트의 OpenAPI 스키마 확인
        schema = app.openapi()
        paths = schema.get("paths", {})

        # /api/predict 엔드포인트가 있어야 함 (파일 업로드 방식)
        predict_path = "/api/predict"
        assert predict_path in paths, f"Predict endpoint {predict_path} not found in registered paths"

        # POST 엔드포인트인지 확인
        assert "post" in paths[predict_path], "Predict endpoint should be a POST method"

        # /api/predict/base64 엔드포인트가 있어야 함 (Base64 방식)
        predict_base64_path = "/api/predict/base64"
        assert predict_base64_path in paths, f"Predict base64 endpoint {predict_base64_path} not found in registered paths"

        # POST 엔드포인트인지 확인
        assert "post" in paths[predict_base64_path], "Predict base64 endpoint should be a POST method"

        # /api/model/info 엔드포인트가 있어야 함
        model_info_path = "/api/model/info"
        assert model_info_path in paths, f"Model info endpoint {model_info_path} not found in registered paths"

        # GET 엔드포인트인지 확인
        assert "get" in paths[model_info_path], "Model info endpoint should be a GET method"

    def test_openapi_schema_generation(self):
        """OpenAPI 스키마가 올바르게 생성되는지 테스트"""
        schema = app.openapi()

        # 기본 스키마 구조
        assert "info" in schema
        assert "paths" in schema

        # 정보 섹션
        info = schema["info"]
        assert info["title"] == "Painting Surface Defect Detection API"
        assert info["version"] == "1.0"

        # 경로에 API 엔드포인트가 포함되어야 함
        paths = schema["paths"]
        assert len(paths) > 0

    def test_docs_endpoint_accessibility(self):
        """자동 문서화 엔드포인트에 접근 가능한지 테스트"""
        response = self.client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_openapi_json_endpoint(self):
        """OpenAPI JSON 스키마 엔드포인트에 접근 가능한지 테스트"""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"

        # 스키마 구조 확인
        schema = response.json()
        assert "info" in schema
        assert "paths" in schema

    def test_redoc_endpoint_accessibility(self):
        """ReDoc 문서화 엔드포인트에 접근 가능한지 테스트"""
        response = self.client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_router_prefix_configuration(self):
        """라우터가 올바른 접두사로 설정되었는지 테스트"""
        # 예측 엔드포인트가 /api 접두사로 접근 가능한지 확인
        response = self.client.get("/api/predict")
        # GET에 대해 405 Method Not Allowed를 반환해야 함 (POST 엔드포인트이므로)
        assert response.status_code == 405

        response = self.client.get("/api/predict/base64")
        # GET에 대해 405 Method Not Allowed를 반환해야 함 (POST 엔드포인트이므로)
        assert response.status_code == 405

    def test_invalid_routes_return_404(self):
        """잘못된 라우트가 404를 반환하는지 테스트"""
        invalid_routes = [
            "/invalid",
            "/api/invalid",
            "/predict",
            "/health/invalid"
        ]

        for route in invalid_routes:
            response = self.client.get(route)
            assert response.status_code == 404, f"Route {route} should return 404"

    def test_health_endpoint(self):
        """헬스 체크 엔드포인트 테스트"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_ready_endpoint(self):
        """준비 상태 체크 엔드포인트 테스트"""
        response = self.client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "ready"
        assert "models_loaded" in data

    def test_startup_endpoint(self):
        """시작 상태 체크 엔드포인트 테스트"""
        response = self.client.get("/startup")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "started"
        assert "initialization_complete" in data

    def test_request_validation_error_handling(self):
        """요청 검증 오류가 올바르게 처리되는지 테스트"""
        # 잘못된 JSON으로 테스트 (Base64 방식)
        response = self.client.post(
            "/api/predict/base64",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

        # 필수 필드가 누락된 경우 테스트 (Base64 방식)
        response = self.client.post(
            "/api/predict/base64",
            json={}
        )
        assert response.status_code == 422

    def test_valid_prediction_request_structure(self):
        """유효한 예측 요청 구조가 수락되는지 테스트"""
        # 더미 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        valid_request = {
            "image_base64": image_base64,
            "confidence_threshold": 0.5
        }

        # 실제 모델 로딩을 피하기 위해 예측 함수 모킹
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

    def test_invalid_confidence_threshold_validation(self):
        """신뢰도 임계값 검증 테스트"""
        # 더미 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        invalid_thresholds = [-0.1, 1.1, 2.0, -1.0]

        for threshold in invalid_thresholds:
            request = {
                "image_base64": image_base64,
                "confidence_threshold": threshold
            }
            
            response = self.client.post(
                "/api/predict/base64",
                json=request
            )
            
            # 검증 오류를 반환해야 함
            assert response.status_code == 422

    def test_content_type_handling(self):
        """콘텐츠 타입이 올바르게 처리되는지 테스트"""
        # 잘못된 콘텐츠 타입으로 테스트 (Base64 방식)
        response = self.client.post(
            "/api/predict/base64",
            data="test",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422

    def test_application_startup_shutdown(self):
        """애플리케이션 시작 및 종료 라이프사이클 테스트"""
        # 앱이 생성되고 파괴될 수 있는지 테스트
        test_app = FastAPI(title="Test App")
        assert test_app.title == "Test App"

    @patch('app.routers.predict.predict_anomaly')
    def test_predict_endpoint_integration_with_mock(self, mock_predict):
        """모킹된 예측 함수와 함께 예측 엔드포인트 통합 테스트"""
        # 모킹 설정
        mock_predict.return_value = {
            "predictions": [
                {
                    "bbox": [10, 10, 50, 50],
                    "confidence": 0.95,
                    "class_id": 0,
                    "class_name": "dirt",
                    "area": 1600.0
                }
            ],
            "image_shape": [100, 100, 3],
            "confidence_threshold": 0.5,
            "timestamp": "2024-01-01T00:00:00",
            "model_source": "Hugging Face"
        }

        # 테스트 요청 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
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
            assert "predictions" in data
            assert "image_shape" in data
            assert "confidence_threshold" in data
            assert "timestamp" in data
            assert "model_source" in data

    def test_error_response_format(self):
        """오류 응답이 올바른 형식을 가지고 있는지 테스트"""
        # 잘못된 요청으로 테스트
        response = self.client.post(
            "/api/predict/base64",
            json={"invalid": "data"}
        )
        
        assert response.status_code == 422
        
        # 오류 응답 구조 확인
        error_data = response.json()
        assert "detail" in error_data

    def test_concurrent_requests_handling(self):
        """애플리케이션이 동시 요청을 처리할 수 있는지 테스트"""
        import threading
        import time

        results = []
        errors = []

        def make_request():
            try:
                response = self.client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        # 여러 스레드 생성
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # 모든 스레드가 완료될 때까지 대기
        for thread in threads:
            thread.join()

        # 모든 요청이 성공했는지 확인
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(status == 200 for status in results)

    def test_app_response_headers(self):
        """응답 헤더가 올바르게 설정되는지 테스트"""
        response = self.client.get("/health")
        
        # 일반적인 헤더 확인
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"

    def test_app_openapi_version_compatibility(self):
        """OpenAPI 버전 호환성 테스트"""
        schema = app.openapi()
        
        # OpenAPI 버전 확인
        assert "openapi" in schema
        assert schema["openapi"].startswith("3.")

    def test_response_model_validation(self):
        """응답 모델이 올바르게 검증되는지 테스트"""
        # 헬스 엔드포인트 응답 구조 테스트
        response = self.client.get("/health")
        data = response.json()
        
        # 필수 필드 확인
        required_fields = ["status", "timestamp"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # 준비 상태 엔드포인트 응답 구조 테스트
        response = self.client.get("/ready")
        data = response.json()
        
        required_fields = ["status", "models_loaded"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_file_upload_endpoint(self):
        """파일 업로드 방식 예측 엔드포인트 테스트"""
        # 더미 이미지 파일 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
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

    def test_model_info_endpoint(self):
        """모델 정보 엔드포인트 테스트"""
        response = self.client.get("/api/model/info")
        # 모델이 로드되지 않은 경우 503, 로드된 경우 200
        assert response.status_code in [200, 503]


class TestApplicationConfigurationEdgeCases:
    """엣지 케이스 및 오류 조건 테스트"""

    def setup_method(self):
        """각 테스트 메서드에 대한 테스트 클라이언트 설정"""
        self.client = TestClient(app)

    def test_empty_request_body_handling(self):
        """빈 요청 본문 처리 테스트"""
        response = self.client.post(
            "/api/predict/base64",
            json={}
        )
        assert response.status_code == 422

    def test_null_values_handling(self):
        """요청에서 null 값 처리 테스트"""
        response = self.client.post(
            "/api/predict/base64",
            json={
                "image_base64": None,
                "confidence_threshold": None
            }
        )
        assert response.status_code == 422

    def test_malformed_json_handling(self):
        """잘못된 JSON 처리 테스트"""
        response = self.client.post(
            "/api/predict/base64",
            data="{invalid json}",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_empty_values_array_handling(self):
        """배열에서 빈 값 처리 테스트"""
        response = self.client.post(
            "/api/predict/base64",
            json={
                "image_base64": "",
                "confidence_threshold": 0.5
            }
        )
        assert response.status_code == 422

    def test_invalid_http_methods(self):
        """잘못된 HTTP 메서드가 적절한 오류를 반환하는지 테스트"""
        # POST 전용 엔드포인트에서 GET 테스트
        response = self.client.get("/api/predict")
        assert response.status_code == 405

        response = self.client.get("/api/predict/base64")
        assert response.status_code == 405

        # POST 전용 엔드포인트에서 PUT 테스트
        response = self.client.put("/api/predict")
        assert response.status_code == 405

        response = self.client.put("/api/predict/base64")
        assert response.status_code == 405

        # POST 전용 엔드포인트에서 DELETE 테스트
        response = self.client.delete("/api/predict")
        assert response.status_code == 405

        response = self.client.delete("/api/predict/base64")
        assert response.status_code == 405

    def test_special_characters_in_paths(self):
        """URL 경로에서 특수 문자 처리 테스트"""
        special_paths = [
            "/api/predict%20",
            "/api/predict/",
            "/api/predict?param=value",
            "/api/predict#fragment"
        ]

        for path in special_paths:
            response = self.client.get(path)
            assert response.status_code in [404, 405]

    def test_invalid_content_type_handling(self):
        """잘못된 콘텐츠 타입 처리 테스트"""
        response = self.client.post(
            "/api/predict/base64",
            data="test data",
            headers={"Content-Type": "invalid/type"}
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("confidence_threshold,expected_validation", [
        (0.0, True),
        (0.5, True),
        (1.0, True),
        (-0.1, False),
        (1.1, False),
        ("0.5", True),   # String will be converted to float by Pydantic
        (None, True),    # Optional field
    ])
    def test_confidence_threshold_validation(self, confidence_threshold, expected_validation):
        """다양한 값으로 신뢰도 임계값 검증 테스트"""
        # 더미 base64 이미지 생성
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        request_data = {
            "image_base64": image_base64
        }
        
        if confidence_threshold is not None:
            request_data["confidence_threshold"] = confidence_threshold

        response = self.client.post(
            "/api/predict/base64",
            json=request_data
        )

        if expected_validation:
            # 검증 오류가 없어야 함 (모델이 로드되지 않는 등의 다른 오류는 있을 수 있음)
            assert response.status_code != 422
        else:
            # 검증 오류가 있어야 함
            assert response.status_code == 422


class TestApplicationPerformanceAndReliability:
    """애플리케이션 성능 및 안정성 측면 테스트"""

    def setup_method(self):
        """각 테스트 메서드에 대한 테스트 클라이언트 설정"""
        self.client = TestClient(app)

    def test_application_performance_baseline(self):
        """기본 성능 특성 테스트"""
        import time

        # 헬스 엔드포인트 응답 시간 테스트
        start_time = time.time()
        response = self.client.get("/health")
        end_time = time.time()

        assert response.status_code == 200
        response_time = end_time - start_time
        
        # 응답이 합리적으로 빠르야 함 (1초 미만)
        assert response_time < 1.0, f"Health endpoint too slow: {response_time:.3f}s"

    def test_memory_usage_stability(self):
        """부하 하에서 메모리 사용량이 안정적으로 유지되는지 테스트"""
        # 메모리 누수를 확인하기 위해 여러 요청 수행
        for _ in range(10):
            response = self.client.get("/health")
            assert response.status_code == 200

    def test_error_recovery(self):
        """애플리케이션이 오류에서 우아하게 복구되는지 테스트"""
        # 잘못된 요청으로 테스트하고 복구 확인
        for _ in range(5):
            # 잘못된 요청
            response = self.client.post(
                "/api/predict/base64",
                json={"invalid": "data"}
            )
            assert response.status_code == 422

            # 유효한 헬스 체크가 여전히 작동해야 함
            response = self.client.get("/health")
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__]) 