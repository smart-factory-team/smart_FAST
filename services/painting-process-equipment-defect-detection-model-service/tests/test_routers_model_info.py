import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, status, HTTPException
from unittest.mock import MagicMock
from app.routers.model_info import router
from app.dependencies import get_config

# 테스트에 사용될 가상의 설정(config)
MOCK_CONFIG_SUCCESS = {
    "model": {"name": "random_forest", "version": "1.0.0"},
    "training": {"epochs": 100},
    "threshold": {"confidence": 0.8},
    "features": {"input_size": 3},
    "other_config_key": "should_be_filtered_out"
}

app = FastAPI()
app.include_router(router, prefix="/model")
client = TestClient(app)

class TestGetModelInfo:
    def test_get_model_info_success_and_filtering(self):
        """정상적인 설정으로 모델 정보 반환 및 필터링 테스트"""
        with TestClient(app) as client:
            app.dependency_overrides[get_config] = lambda: MOCK_CONFIG_SUCCESS
            response = client.get("/model")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "other_config_key" not in data
            assert data["model"]["name"] == "random_forest"
        app.dependency_overrides = {}

    def test_get_model_info_with_none_values(self):
        """설정 파일에 키는 있지만 값이 None인 경우, 올바르게 처리되는지 테스트"""
        mock_config = {"model": None, "training": None, "threshold": None, "features": None}
        with TestClient(app) as client:
            app.dependency_overrides[get_config] = lambda: mock_config
            response = client.get("/model")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["model"] is None
            assert data["training"] is None
            assert data["threshold"] is None
        app.dependency_overrides = {}

    def test_get_model_info_dependency_fails(self):
        """get_config 의존성에서 HTTPException이 발생했을 때 500 에러를 반환하는지 테스트"""
        def mock_get_config_fail():
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Configuration not loaded")
        
        with TestClient(app) as client:
            app.dependency_overrides[get_config] = mock_get_config_fail
            response = client.get("/model")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Configuration not loaded" in response.json()["detail"]
        app.dependency_overrides = {}