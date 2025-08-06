import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, call
from datetime import datetime
import json

# 테스트 대상 라우터 및 의존성 함수 임포트
from app.routers.predict import router
from app.dependencies import get_config, get_model, get_explainer
from app.services.utils import IssueLogInput

# --- Mock 데이터 설정 ---

# 실제 model_config.yaml 구조와 거의 동일하게 Mock 설정 생성
MOCK_CONFIG = {
    "model": {"name": "test_model"},
    "threshold": {
        "thickness_error_threshold": 0.1,
        "shap_threshold": 2.0,
    },
    "features": {
        "input_columns": ["voltage", "current", "temper"],
        "target_column": "thick",
    },
    "validation": {
        "voltage_range": [200.0, 300.0],
        "current_range": [10.0, 80.0],
        "temperature_range": [30.0, 40.0],
    },
    "logs": {
        "log_issue_codes": {
            "voltage_invalid": "PAINT-EQ-VOL-INVALID",
            "current_invalid": "PAINT-EQ-CUR-INVALID",
            "temperature_invalid": "PAINT-EQ-TEM-INVALID",
            "machine_issue": "PAINT-EQ-MCH",
        },
        "issue_map": {
            "voltage": "PAINT-EQ-VOL",
            "current": "PAINT-EQ-CUR",
            "temper": "PAINT-EQ-TEM",
        },
        "file_path": "logs/test_issue_logs.jsonl",
    },
    "file_upload": {
        "column_mapping": {
            "machineId": "machineId",
            "timeStamp": "timeStamp",
            "Thick": "thick",
            "PT_jo_V_1": "voltage",
            "PT_jo_A_Main_1": "current",
            "PT_jo_TP": "temper",
        }
    },
}

# analyze_issue_log_api가 반환할 Mock 결과 데이터
MOCK_ISSUE_RESULT = {
    "machineId": "MCH-001",
    "timeStamp": datetime.now().isoformat(),
    "thick": 20.5,
    "voltage": 250.0,
    "current": 75.0,
    "temper": 35.0,
    "issue": "PAINT-EQ-VOL-HIGH-2025-08-04T12:00:00",
    "isSolved": False,
}

# 정상 입력 데이터
MOCK_INPUT_DATA = IssueLogInput(
    machineId="MCH-001",
    timeStamp=datetime.now(),
    thick=20.5,
    voltage=250.0,
    current=75.0,
    temper=35.0,
)

# --- 테스트 환경 설정 ---

# 의존성을 오버라이드한 테스트용 FastAPI 앱과 클라이언트 생성
@pytest.fixture(scope="module")
def test_client():
    app = FastAPI()
    app.include_router(router, prefix="/predict")

    # 의존성 오버라이드
    app.dependency_overrides[get_config] = lambda: MOCK_CONFIG
    app.dependency_overrides[get_model] = lambda: MagicMock()
    app.dependency_overrides[get_explainer] = lambda: MagicMock()

    with TestClient(app) as client:
        yield client

# --- 테스트 케이스 ---

class TestPredictEndpoint:
    """/predict 엔드포인트 테스트"""

    @patch("app.routers.predict.run_analysis", return_value=MOCK_ISSUE_RESULT)
    @patch("app.routers.predict.save_issue_log")
    def test_predict_with_issue(self, mock_save, mock_analysis, test_client):
        """이슈가 감지되어 200 OK와 예측 결과를 반환하고 로그를 저장하는 경우"""
        response = test_client.post("/predict/", json=json.loads(MOCK_INPUT_DATA.model_dump_json()))
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "predictions" in data
        assert data["predictions"][0]["issue"] == MOCK_ISSUE_RESULT["issue"]
        mock_analysis.assert_called_once()
        mock_save.assert_called_once()

    @patch("app.routers.predict.run_analysis", return_value=None)
    @patch("app.routers.predict.save_issue_log")
    def test_predict_no_issue(self, mock_save, mock_analysis, test_client):
        """이슈가 감지되지 않아 204 No Content를 반환하는 경우"""
        response = test_client.post("/predict/", json=json.loads(MOCK_INPUT_DATA.model_dump_json()))
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_analysis.assert_called_once()
        mock_save.assert_not_called()

    def test_predict_invalid_input(self, test_client):
        """입력 데이터가 유효성 검사를 통과하지 못해 422 에러를 반환하는 경우"""
        invalid_data = MOCK_INPUT_DATA.model_dump()
        del invalid_data["voltage"]  # 필수 필드 제거
        
        # Pydantic 모델의 직렬화 이슈를 피하기 위해 datetime을 문자열로 변환
        invalid_data["timeStamp"] = invalid_data["timeStamp"].isoformat()

        response = test_client.post("/predict/", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_out_of_range_input(self, test_client):
        """입력 데이터가 정상 범위를 벗어났을 때, 올바른 이슈 코드를 반환하는지 테스트"""
        # 전압(voltage)이 정상 범위를 벗어나는 데이터 생성
        out_of_range_data = MOCK_INPUT_DATA.model_dump()
        out_of_range_data["voltage"] = 350.0  # MOCK_CONFIG의 범위를 벗어남

        # Pydantic 모델의 직렬화 이슈를 피하기 위해 datetime을 문자열로 변환
        out_of_range_data["timeStamp"] = out_of_range_data["timeStamp"].isoformat()

        response = test_client.post("/predict/", json=out_of_range_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["predictions"][0]["issue"] == MOCK_CONFIG["logs"]["log_issue_codes"]["voltage_invalid"]

class TestPredictFileEndpoint:
    """/predict/file 엔드포인트 테스트"""

    @patch("app.routers.predict.run_analysis", side_effect=[MOCK_ISSUE_RESULT, None])
    @patch("app.routers.predict.save_issue_log")
    def test_predict_file_success(self, mock_save, mock_analysis, test_client):
        """정상 CSV 파일 업로드 시, 스트리밍으로 처리하고 결과를 반환하는지 테스트"""
        # model_config.yaml의 column_mapping "key"를 헤더로 사용
        csv_data = (
            "machineId,timeStamp,Thick,PT_jo_V_1,PT_jo_A_Main_1,PT_jo_TP\n"
            "MCH-001,2025-08-04T12:00:00,20.5,250.0,75.0,35.0\n"
            "MCH-002,2025-08-04T12:01:00,20.0,260.0,70.0,36.0"
        )
        
        response = test_client.post(
            "/predict/file",
            files={"file": ("test.csv", csv_data, "text/csv")}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["filename"] == "test.csv"
        assert len(data["predictions"]) == 1  # 이슈가 있는 1건만 결과에 포함
        assert data["predictions"][0]["issue"] == MOCK_ISSUE_RESULT["issue"]
        
        # run_analysis가 2번(각 행마다) 호출되었는지 확인
        assert mock_analysis.call_count == 2
        # save_issue_log가 1번(이슈가 있는 경우만) 호출되었는지 확인
        mock_save.assert_called_once()

    def test_predict_file_invalid_type(self, test_client):
        """CSV가 아닌 다른 형식의 파일 업로드 시 400 에러를 반환하는지 테스트"""
        response = test_client.post(
            "/predict/file",
            files={"file": ("test.txt", "some text data", "text/plain")}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "CSV 파일만 허용됩니다" in response.json()["detail"]

    def test_predict_file_missing_columns(self, test_client):
        """필수 컬럼이 누락된 CSV 파일 업로드 시 400 에러를 반환하는지 테스트"""
        # "PT_jo_V_1" (voltage) 컬럼 누락
        csv_data = (
            "machineId,timeStamp,Thick,PT_jo_A_Main_1,PT_jo_TP\n"
            "MCH-001,2025-08-04T12:00:00,20.5,75.0,35.0"
        )
        
        response = test_client.post(
            "/predict/file",
            files={"file": ("test_missing_col.csv", csv_data, "text/csv")}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "필수 컬럼이 누락되었습니다" in response.json()["detail"]
