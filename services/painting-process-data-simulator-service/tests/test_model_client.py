import pytest
import httpx
import respx
from typing import Dict, Any

from app.services.model_client import model_client
from app.config.settings import settings

# 모든 테스트는 비동기로 실행되도록 표시
pytestmark = pytest.mark.asyncio

# 테스트에 사용할 모델 서비스 URL
SERVICE_NAME = "painting-process-equipment"
BASE_URL = settings.model_services.get(SERVICE_NAME)
PREDICT_URL = f"{BASE_URL}/predict/"

# 테스트용 입력 데이터
@pytest.fixture
def sample_input_data() -> Dict[str, Any]:
    return {
        "machineId": "TEST-MCH-01",
        "timeStamp": "2025-08-06T12:00:00",
        "thick": 25.0,
        "voltage": 300.0,
        "current": 80.0,
        "temper": 35.0,
        "issue": "",
        "isSolved": False
    }

async def test_predict_painting_issue_success_with_issue(sample_input_data: Dict[str, Any], respx_mock):
    """모델이 이슈를 감지했을 때 (200 OK) 응답을 올바르게 처리하는지 테스트"""
    # given: 모델 서비스가 200 OK와 함께 이슈 데이터를 반환하도록 모킹
    mock_response = {
        "predictions": [{
            "issue": "PAINT-EQ-VOL-HIGH"
        }]
    }
    respx_mock.post(PREDICT_URL).mock(return_value=httpx.Response(200, json=mock_response))

    # when: predict_painting_issue 함수 호출
    async with httpx.AsyncClient() as client:
        result = await model_client.predict_painting_issue(sample_input_data, client)

    # then: 반환된 결과에 machineId와 timeStamp가 포함되고, issue가 올바른지 확인
    assert result is not None
    assert result["issue"] == "PAINT-EQ-VOL-HIGH"
    assert result["machineId"] == sample_input_data["machineId"]
    assert result["timeStamp"] == sample_input_data["timeStamp"]

async def test_predict_painting_issue_success_no_issue(sample_input_data: Dict[str, Any], respx_mock):
    """모델이 정상으로 판단했을 때 (204 No Content) 응답을 올바르게 처리하는지 테스트"""
    # given: 모델 서비스가 204 No Content를 반환하도록 모킹
    respx_mock.post(PREDICT_URL).mock(return_value=httpx.Response(204))

    # when: predict_painting_issue 함수 호출
    async with httpx.AsyncClient() as client:
        result = await model_client.predict_painting_issue(sample_input_data, client)

    # then: 결과가 None인지 확인
    assert result is None

async def test_predict_painting_issue_http_error(sample_input_data: Dict[str, Any], respx_mock):
    """모델 서비스가 HTTP 오류(500)를 반환했을 때 None을 반환하는지 테스트"""
    # given: 모델 서비스가 500 Internal Server Error를 반환하도록 모킹
    respx_mock.post(PREDICT_URL).mock(return_value=httpx.Response(500))

    # when: predict_painting_issue 함수 호출
    async with httpx.AsyncClient() as client:
        result = await model_client.predict_painting_issue(sample_input_data, client)

    # then: 결과가 None인지 확인 (최대 재시도 후)
    assert result is None

async def test_predict_painting_issue_timeout_error(sample_input_data: Dict[str, Any], respx_mock):
    """모델 서비스 연결 시 타임아웃이 발생했을 때 None을 반환하는지 테스트"""
    # given: 모델 서비스 연결 시 Timeout 예외가 발생하도록 모킹
    respx_mock.post(PREDICT_URL).mock(side_effect=httpx.TimeoutException("Timeout error"))

    # when: predict_painting_issue 함수 호출
    async with httpx.AsyncClient() as client:
        result = await model_client.predict_painting_issue(sample_input_data, client)

    # then: 결과가 None인지 확인 (최대 재시도 후)
    assert result is None


