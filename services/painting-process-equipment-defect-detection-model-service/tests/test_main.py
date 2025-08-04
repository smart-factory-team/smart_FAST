import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.main import app

def test_startup_success():
    """애플리케이션 시작 시 리소스 로딩이 성공적으로 이루어지는지 테스트"""
    with patch('app.main.load_resources', new=AsyncMock(return_value=None)) as mock_load:
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            mock_load.assert_awaited_once()

def test_startup_failure():
    """애플리케이션 시작 시 리소스 로딩이 실패하면 앱이 시작되지 않는지 테스트"""
    # 목업 함수가 RuntimeError를 발생시키도록 수정
    with patch('app.main.load_resources', new=AsyncMock(side_effect=RuntimeError("리소스 로딩 실패"))):
        with pytest.raises(RuntimeError):
            with TestClient(app):
                pass