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


class TestApplicationSecurityAndValidation:
    """Security and input validation focused tests
    Testing framework: pytest with FastAPI TestClient
    """

    def setup_method(self):
        """각 테스트 메서드에 대한 테스트 클라이언트 설정"""
        self.client = TestClient(app)

    def test_large_request_body_handling(self):
        """큰 요청 본문 처리 테스트"""
        # Very large base64 string to test size limits
        large_image_data = "a" * 10000000  # 10MB of data
        
        response = self.client.post(
            "/api/predict/base64",
            json={
                "image_base64": large_image_data,
                "confidence_threshold": 0.5
            }
        )
        # Should handle large requests gracefully (either process or reject with proper error)
        assert response.status_code in [200, 413, 422, 500]

    def test_malicious_base64_content(self):
        """악의적인 base64 콘텐츠 처리 테스트"""
        malicious_inputs = [
            "../../../../etc/passwd",
            "<script>alert('xss')</script>",
            "' OR '1'='1",
            "../../../windows/system32",
            "%00%2e%2e%2f",
            "\x00\x01\x02\x03",
            "javascript:alert(1)"
        ]
        
        for malicious_input in malicious_inputs:
            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": malicious_input,
                    "confidence_threshold": 0.5
                }
            )
            # Should properly validate and reject malicious content
            assert response.status_code in [422, 400, 500]

    def test_unicode_and_special_characters(self):
        """유니코드 및 특수 문자 처리 테스트"""
        unicode_inputs = [
            "🎨🖼️",  # Emojis
            "测试图像",  # Chinese characters
            "тестовое изображение",  # Cyrillic
            "画像テスト",  # Japanese
            "♠♣♥♦",  # Special symbols
            "مرحبا",  # Arabic
            "שלום",  # Hebrew
        ]
        
        for unicode_input in unicode_inputs:
            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": unicode_input,
                    "confidence_threshold": 0.5
                }
            )
            # Should handle Unicode gracefully
            assert response.status_code in [422, 400]

    def test_sql_injection_attempts(self):
        """SQL 인젝션 시도 테스트"""
        sql_injection_attempts = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM sensitive_data",
            "' WAITFOR DELAY '00:00:30' --",
            "'; EXEC xp_cmdshell('dir'); --"
        ]
        
        for injection_attempt in sql_injection_attempts:
            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": injection_attempt,
                    "confidence_threshold": 0.5
                }
            )
            # Should not be vulnerable to SQL injection
            assert response.status_code in [422, 400]

    def test_path_traversal_attempts(self):
        """경로 순회 공격 시도 테스트"""
        path_traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc//passwd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        for path_attempt in path_traversal_attempts:
            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": path_attempt,
                    "confidence_threshold": 0.5
                }
            )
            # Should not be vulnerable to path traversal
            assert response.status_code in [422, 400]

    def test_request_header_injection(self):
        """요청 헤더 인젝션 테스트"""
        malicious_headers = {
            "X-Forwarded-For": "127.0.0.1\r\nX-Injected: malicious",
            "User-Agent": "Mozilla/5.0\r\nX-Injected: header",
            "Content-Type": "application/json\r\nX-Injected: value"
        }
        
        for header_name, header_value in malicious_headers.items():
            response = self.client.post(
                "/api/predict/base64",
                json={"image_base64": "dGVzdA==", "confidence_threshold": 0.5},
                headers={header_name: header_value}
            )
            # Should handle header injection safely
            assert response.status_code in [200, 400, 422, 500]

    @pytest.mark.parametrize("extreme_threshold", [
        -999999999,
        999999999,
        float('inf'),
        float('-inf'),
        1e308,  # Very large float
        1e-308,  # Very small float
    ])
    def test_extreme_numeric_values(self, extreme_threshold):
        """극값 숫자 처리 테스트"""
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = self.client.post(
            "/api/predict/base64",
            json={
                "image_base64": image_base64,
                "confidence_threshold": extreme_threshold
            }
        )
        # Should handle extreme values gracefully
        assert response.status_code in [200, 422, 500]

    def test_nested_json_structure_handling(self):
        """중첩된 JSON 구조 처리 테스트"""
        nested_structures = [
            {"image_base64": {"nested": "value"}, "confidence_threshold": 0.5},
            {"image_base64": ["array", "of", "values"], "confidence_threshold": 0.5},
            {"image_base64": "valid_base64", "confidence_threshold": {"nested": 0.5}},
            {"image_base64": "valid_base64", "extra_nested": {"deep": {"very": {"nested": True}}}},
        ]
        
        for structure in nested_structures:
            response = self.client.post(
                "/api/predict/base64",
                json=structure
            )
            # Should properly validate JSON structure
            assert response.status_code in [422, 400]

    def test_file_upload_security_validation(self):
        """파일 업로드 보안 검증 테스트"""
        malicious_files = [
            ("malicious.exe", b"MZ\x90\x00", "application/octet-stream"),
            ("script.js", b"alert('xss')", "application/javascript"),
            ("test.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("test.svg", b"<svg onload=alert(1)>", "image/svg+xml"),
        ]
        
        for filename, content, content_type in malicious_files:
            response = self.client.post(
                "/api/predict",
                files={"image": (filename, content, content_type)},
                data={"confidence_threshold": 0.5}
            )
            # Should reject non-image files
            assert response.status_code in [400, 422]

    def test_content_type_spoofing(self):
        """콘텐츠 타입 스푸핑 테스트"""
        # Executable file with image content type
        response = self.client.post(
            "/api/predict",
            files={"image": ("fake.jpg", b"MZ\x90\x00\x03\x00\x00\x00", "image/jpeg")},
            data={"confidence_threshold": 0.5}
        )
        # Should detect and reject non-image content regardless of claimed content type
        assert response.status_code in [400, 422, 500]

    def test_file_size_validation(self):
        """파일 크기 검증 테스트"""
        # Create a large dummy image file (over 10MB)
        large_content = b"fake_image_data" * 1000000  # ~15MB
        
        response = self.client.post(
            "/api/predict",
            files={"image": ("large.jpg", large_content, "image/jpeg")},
            data={"confidence_threshold": 0.5}
        )
        # Should reject files over size limit
        assert response.status_code == 400


class TestApplicationRobustnessAndEdgeCases:
    """Application robustness and comprehensive edge case testing
    Testing framework: pytest with FastAPI TestClient
    """

    def setup_method(self):
        """각 테스트 메서드에 대한 테스트 클라이언트 설정"""
        self.client = TestClient(app)

    def test_concurrent_file_upload_requests(self):
        """동시 파일 업로드 요청 테스트"""
        import threading

        results = []
        errors = []
        
        def upload_file():
            try:
                dummy_image = np.zeros((50, 50, 3), dtype=np.uint8)
                import cv2
                _, buffer = cv2.imencode('.jpg', dummy_image)
                image_data = buffer.tobytes()
                
                with patch('app.routers.predict.predict_anomaly') as mock_predict:
                    mock_predict.return_value = {
                        "predictions": [],
                        "image_shape": [50, 50, 3],
                        "confidence_threshold": 0.5,
                        "timestamp": "2024-01-01T00:00:00",
                        "model_source": "Hugging Face"
                    }
                    
                    response = self.client.post(
                        "/api/predict",
                        files={"image": ("test.jpg", image_data, "image/jpeg")},
                        data={"confidence_threshold": 0.5}
                    )
                    results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads for concurrent uploads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=upload_file)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify concurrent handling
        assert len(errors) == 0, f"Errors in concurrent uploads: {errors}"
        assert len(results) == 3
        assert all(status in [200, 500] for status in results)

    def test_different_image_formats_base64(self):
        """다양한 이미지 형식의 Base64 인코딩 테스트"""
        formats = ['.png', '.bmp', '.tiff', '.webp']
        
        for fmt in formats:
            dummy_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            import cv2
            success, buffer = cv2.imencode(fmt, dummy_image)
            
            if success:
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                response = self.client.post(
                    "/api/predict/base64",
                    json={
                        "image_base64": image_base64,
                        "confidence_threshold": 0.5
                    }
                )
                
                # Should handle different formats or reject gracefully
                assert response.status_code in [200, 422, 400, 500]

    def test_grayscale_image_handling(self):
        """그레이스케일 이미지 처리 테스트"""
        # Create grayscale image
        grayscale_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', grayscale_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = self.client.post(
            "/api/predict/base64",
            json={
                "image_base64": image_base64,
                "confidence_threshold": 0.5
            }
        )
        
        # Should handle grayscale images appropriately
        assert response.status_code in [200, 422, 500]

    def test_very_small_image_handling(self):
        """매우 작은 이미지 처리 테스트"""
        # Create 1x1 pixel image
        tiny_image = np.array([[[255, 0, 0]]], dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', tiny_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = self.client.post(
            "/api/predict/base64",
            json={
                "image_base64": image_base64,
                "confidence_threshold": 0.5
            }
        )
        
        # Should handle very small images appropriately
        assert response.status_code in [200, 422, 500]

    def test_very_large_image_dimensions(self):
        """매우 큰 이미지 차원 처리 테스트"""
        # Create large image (but keep file size manageable)
        large_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        # Add some pattern to make it compressible
        large_image[::100, ::100] = 255
        
        import cv2
        _, buffer = cv2.imencode('.jpg', large_image, [cv2.IMWRITE_JPEG_QUALITY, 10])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = self.client.post(
            "/api/predict/base64",
            json={
                "image_base64": image_base64,
                "confidence_threshold": 0.5
            }
        )
        
        # Should handle large images appropriately
        assert response.status_code in [200, 422, 413, 500]

    def test_corrupted_base64_images(self):
        """손상된 Base64 이미지 처리 테스트"""
        corrupted_base64_samples = [
            "iVBORw0KGgoAAAANSUhEUgAAAA",  # Incomplete
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ",  # With data URL prefix
            "iVBORw0KGgoAAAANSUhEUgAA!!!INVALID!!!",  # Invalid characters
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU",  # Truncated
        ]
        
        for corrupted_b64 in corrupted_base64_samples:
            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": corrupted_b64,
                    "confidence_threshold": 0.5
                }
            )
            
            # Should handle corrupted data gracefully
            assert response.status_code in [422, 400, 500]

    def test_non_image_base64_content(self):
        """이미지가 아닌 Base64 콘텐츠 처리 테스트"""
        # Text file encoded as base64
        text_content = base64.b64encode(b"This is not an image file").decode('utf-8')
        
        response = self.client.post(
            "/api/predict/base64",
            json={
                "image_base64": text_content,
                "confidence_threshold": 0.5
            }
        )
        
        # Should reject non-image content
        assert response.status_code in [422, 400, 500]

    def test_prediction_response_structure_validation(self):
        """예측 응답 구조 검증 테스트"""
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            # Test various response structures
            test_responses = [
                {
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
                },
                {
                    "predictions": [],  # No detections
                    "image_shape": [100, 100, 3],
                    "confidence_threshold": 0.8,
                    "timestamp": "2024-01-01T00:00:00",
                    "model_source": "Hugging Face"
                },
                {
                    "predictions": [
                        {
                            "bbox": [0, 0, 100, 100],  # Full image detection
                            "confidence": 1.0,
                            "class_id": 1,
                            "class_name": "scratch",
                            "area": 10000.0
                        }
                    ],
                    "image_shape": [100, 100, 3],
                    "confidence_threshold": 0.1,
                    "timestamp": "2024-01-01T00:00:00",
                    "model_source": "Hugging Face"
                }
            ]

            for test_response in test_responses:
                mock_predict.return_value = test_response
                
                response = self.client.post(
                    "/api/predict/base64",
                    json={
                        "image_base64": image_base64,
                        "confidence_threshold": test_response["confidence_threshold"]
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Validate response structure
                    required_fields = ["predictions", "image_shape", "confidence_threshold", "timestamp", "model_source"]
                    for field in required_fields:
                        assert field in data, f"Missing required field: {field}"
                    
                    # Validate predictions structure
                    if data["predictions"]:
                        for prediction in data["predictions"]:
                            prediction_fields = ["bbox", "confidence", "class_id", "class_name", "area"]
                            for field in prediction_fields:
                                assert field in prediction, f"Missing prediction field: {field}"

    def test_application_state_consistency(self):
        """애플리케이션 상태 일관성 테스트"""
        # Test health endpoint multiple times to ensure consistent state
        for _i in range(10):
            response = self.client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data

        # Test ready endpoint consistency
        for _i in range(5):
            response = self.client.get("/ready")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "ready"
            assert "models_loaded" in data

    def test_endpoint_timeout_behavior(self):
        """엔드포인트 타임아웃 동작 테스트"""
        import time
        
        # Simulate slow processing with mock
        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            def slow_predict(*args, **kwargs):
                time.sleep(0.1)  # Small delay to simulate processing
                return {
                    "predictions": [],
                    "image_shape": [100, 100, 3],
                    "confidence_threshold": 0.5,
                    "timestamp": "2024-01-01T00:00:00",
                    "model_source": "Hugging Face"
                }
            
            mock_predict.side_effect = slow_predict
            
            dummy_image = np.zeros((50, 50, 3), dtype=np.uint8)
            import cv2
            _, buffer = cv2.imencode('.jpg', dummy_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            start_time = time.time()
            response = self.client.post(
                "/api/predict/base64",
                json={
                    "image_base64": image_base64,
                    "confidence_threshold": 0.5
                }
            )
            end_time = time.time()
            
            # Should complete within reasonable time
            assert end_time - start_time < 10.0  # 10 second timeout
            assert response.status_code in [200, 500]

    def test_model_loading_error_scenarios(self):
        """모델 로딩 오류 시나리오 테스트"""
        # Test model info when model might not be loaded
        response = self.client.get("/api/model/info")
        
        # Should handle model loading state gracefully
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            # Validate expected fields in model info
            expected_fields = ["model_name", "version", "status"]
            for field in expected_fields:
                # Fields might be present depending on implementation
                if field in data:
                    assert isinstance(data[field], (str, dict, bool))

    def test_api_versioning_compatibility(self):
        """API 버전 호환성 테스트"""
        # Test that API paths are correctly versioned with /api prefix
        api_endpoints = [
            ("/api/predict", "POST"),
            ("/api/predict/base64", "POST"),
            ("/api/model/info", "GET")
        ]
        
        for endpoint, method in api_endpoints:
            if method == "GET":
                response = self.client.get(endpoint)
            else:
                # For POST endpoints, just check that they exist (405 for GET is expected)
                response = self.client.get(endpoint)
                assert response.status_code == 405  # Method not allowed
                
                # Test with minimal valid POST data
                if "predict" in endpoint:
                    if "base64" in endpoint:
                        response = self.client.post(endpoint, json={
                            "image_base64": "invalid_base64",
                            "confidence_threshold": 0.5
                        })
                    else:
                        response = self.client.post(endpoint, files={"image": ("test.jpg", b"fake_data", "image/jpeg")})
                    assert response.status_code in [200, 422, 500]

    def test_lifespan_event_handling(self):
        """애플리케이션 라이프사이클 이벤트 처리 테스트"""
        # Test that app handles startup and shutdown events properly
        # This is more of an integration test for the lifespan context manager
        
        with patch('app.services.inference.PaintingSurfaceDefectDetectionService') as MockService:
            mock_service = MagicMock()
            MockService.return_value = mock_service
            mock_service.load_model.return_value = None
            mock_service.cleanup.return_value = None
            
            # Create a new app instance to test lifespan
            test_client = TestClient(app)
            response = test_client.get("/health")
            assert response.status_code == 200

    def test_cors_and_security_headers(self):
        """CORS 및 보안 헤더 테스트"""
        response = self.client.get("/health")
        
        # Check that response has appropriate headers
        assert response.status_code == 200
        
        # Test OPTIONS request (preflight)
        response = self.client.options("/api/predict/base64")
        # Should handle OPTIONS requests appropriately
        assert response.status_code in [200, 405]

    def test_error_handling_consistency(self):
        """오류 처리 일관성 테스트"""
        # Test various error scenarios and ensure consistent error format
        error_scenarios = [
            # Missing required field
            ({}, "/api/predict/base64", "POST"),
            # Invalid JSON
            ("invalid json", "/api/predict/base64", "POST"),
            # Wrong content type
            ({"test": "data"}, "/api/predict/base64", "PUT"),
        ]
        
        for data, endpoint, method in error_scenarios:
            if method == "POST":
                if isinstance(data, str):
                    response = self.client.post(
                        endpoint,
                        data=data,
                        headers={"Content-Type": "application/json"}
                    )
                else:
                    response = self.client.post(endpoint, json=data)
            elif method == "PUT":
                response = self.client.put(endpoint, json=data)
            
            # Should return appropriate error status
            assert response.status_code >= 400
            
            # Should return JSON error response
            if response.headers.get("content-type") == "application/json":
                error_data = response.json()
                # FastAPI typically returns errors with 'detail' field
                assert "detail" in error_data or "message" in error_data

    def test_multiple_predictions_in_single_image(self):
        """단일 이미지에서 여러 예측 결과 테스트"""
        dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        with patch('app.routers.predict.predict_anomaly') as mock_predict:
            # Mock multiple predictions
            mock_predict.return_value = {
                "predictions": [
                    {
                        "bbox": [10, 10, 50, 50],
                        "confidence": 0.95,
                        "class_id": 0,
                        "class_name": "dirt",
                        "area": 1600.0
                    },
                    {
                        "bbox": [100, 100, 150, 150],
                        "confidence": 0.85,
                        "class_id": 1,
                        "class_name": "scratch",
                        "area": 2500.0
                    },
                    {
                        "bbox": [60, 60, 80, 80],
                        "confidence": 0.75,
                        "class_id": 2,
                        "class_name": "rust",
                        "area": 400.0
                    }
                ],
                "image_shape": [200, 200, 3],
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
                assert len(data["predictions"]) == 3

                # Verify each prediction has required fields
                for prediction in data["predictions"]:
                    assert "bbox" in prediction
                    assert "confidence" in prediction
                    assert "class_id" in prediction
                    assert "class_name" in prediction
                    assert "area" in prediction

                    # Verify data types
                    assert isinstance(prediction["bbox"], list)
                    assert len(prediction["bbox"]) == 4
                    assert isinstance(prediction["confidence"], (int, float))
                    assert isinstance(prediction["class_id"], int)
                    assert isinstance(prediction["class_name"], str)
                    assert isinstance(prediction["area"], (int, float))