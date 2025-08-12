import pytest
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException
from app.routers.simulator_router import get_simulator_status, start_simulator, stop_simulator, get_recent_logs
from app.services.scheduler_service import simulator_scheduler
from app.services.model_client import painting_surface_model_client


class TestSimulatorRouter:
    """시뮬레이터 라우터 테스트"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        self.test_log_dir = "test-logs"
        self.test_log_file = os.path.join(self.test_log_dir, "test_log.json")
        
        # 테스트 로그 디렉토리 생성
        os.makedirs(self.test_log_dir, exist_ok=True)

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리"""
        # 테스트 로그 파일 정리
        if os.path.exists(self.test_log_dir):
            import shutil
            shutil.rmtree(self.test_log_dir)

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.simulator_scheduler')
    @patch('app.routers.simulator_router.painting_surface_model_client')
    async def test_get_simulator_status_success(self, mock_model_client, mock_scheduler):
        """시뮬레이터 상태 조회 성공 테스트"""
        # Mock 스케줄러 상태
        mock_scheduler.is_running = True
        mock_scheduler.get_status.return_value = {
            "running": True,
            "started_at": "2024-01-01T00:00:00",
            "jobs": ["data_simulation"]
        }
        
        # Mock 모델 클라이언트 헬스 체크
        mock_model_client.health_check = AsyncMock(return_value=True)
        
        # 상태 조회
        response = await get_simulator_status()
        
        assert response["running"] is True
        assert response["started_at"] == "2024-01-01T00:00:00"
        assert response["jobs"] == ["data_simulation"]
        assert response["painting_surface_service_health"] is True

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.simulator_scheduler')
    @patch('app.routers.simulator_router.painting_surface_model_client')
    async def test_get_simulator_status_not_running(self, mock_model_client, mock_scheduler):
        """시뮬레이터가 실행 중이 아닌 경우 테스트"""
        # Mock 스케줄러 상태
        mock_scheduler.is_running = False
        mock_scheduler.get_status.return_value = {
            "running": False,
            "started_at": None,
            "jobs": []
        }
        
        # 상태 조회
        response = await get_simulator_status()
        
        assert response["running"] is False
        assert response["started_at"] is None
        assert response["jobs"] == []
        # 실행 중이 아니므로 헬스 체크가 호출되지 않음
        assert "painting_surface_service_health" not in response

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.simulator_scheduler')
    async def test_start_simulator_success(self, mock_scheduler):
        """시뮬레이터 시작 성공 테스트"""
        # Mock 스케줄러 시작
        mock_scheduler.start = AsyncMock()
        mock_scheduler.get_status.return_value = {
            "running": True,
            "started_at": "2024-01-01T00:00:00",
            "jobs": ["data_simulation"]
        }
        
        # 시뮬레이터 시작
        response = await start_simulator()
        
        assert response["message"] == "시뮬레이터가 시작되었습니다."
        assert response["status"]["running"] is True
        mock_scheduler.start.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.simulator_scheduler')
    async def test_start_simulator_failure(self, mock_scheduler):
        """시뮬레이터 시작 실패 테스트"""
        # Mock 스케줄러 시작 실패
        mock_scheduler.start = AsyncMock(side_effect=Exception("Start failed"))
        
        # 시뮬레이터 시작 시도 (예외 발생)
        with pytest.raises(HTTPException) as exc_info:
            await start_simulator()
        
        assert exc_info.value.status_code == 500
        assert "시뮬레이터 시작 실패" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.simulator_scheduler')
    async def test_stop_simulator_success(self, mock_scheduler):
        """시뮬레이터 중지 성공 테스트"""
        # Mock 스케줄러 중지
        mock_scheduler.stop = AsyncMock()
        mock_scheduler.get_status.return_value = {
            "running": False,
            "started_at": None,
            "jobs": []
        }
        
        # 시뮬레이터 중지
        response = await stop_simulator()
        
        assert response["message"] == "시뮬레이터가 중지되었습니다."
        assert response["status"]["running"] is False
        mock_scheduler.stop.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.simulator_scheduler')
    async def test_stop_simulator_failure(self, mock_scheduler):
        """시뮬레이터 중지 실패 테스트"""
        # Mock 스케줄러 중지 실패
        mock_scheduler.stop = AsyncMock(side_effect=Exception("Stop failed"))
        
        # 시뮬레이터 중지 시도 (예외 발생)
        with pytest.raises(HTTPException) as exc_info:
            await stop_simulator()
        
        assert exc_info.value.status_code == 500
        assert "시뮬레이터 중지 실패" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.settings')
    async def test_get_recent_logs_success(self, mock_settings):
        """최근 로그 조회 성공 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "test_log.json"
        
        # 테스트 로그 파일 생성
        test_logs = [
            {"timestamp": "2024-01-01T00:00:00", "service": "test1", "status": "normal"},
            {"timestamp": "2024-01-01T00:01:00", "service": "test2", "status": "anomaly"},
            {"timestamp": "2024-01-01T00:02:00", "service": "test3", "status": "normal"}
        ]
        
        with open(self.test_log_file, 'w', encoding='utf-8') as f:
            for log in test_logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')
        
        # 최근 로그 조회
        response = await get_recent_logs()
        
        assert "logs" in response
        assert "total_count" in response
        assert len(response["logs"]) == 3
        assert response["total_count"] == 3
        
        # 로그 내용 확인
        assert response["logs"][0]["service"] == "test1"
        assert response["logs"][1]["service"] == "test2"
        assert response["logs"][2]["service"] == "test3"

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.settings')
    async def test_get_recent_logs_no_file(self, mock_settings):
        """로그 파일이 없는 경우 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "nonexistent.json"
        
        # 최근 로그 조회
        response = await get_recent_logs()
        
        assert response["logs"] == []
        assert response["message"] == "로그 파일이 없습니다."

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.settings')
    async def test_get_recent_logs_invalid_json(self, mock_settings):
        """잘못된 JSON 형식의 로그 파일 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "test_log.json"
        
        # 잘못된 JSON 형식의 로그 파일 생성
        with open(self.test_log_file, 'w', encoding='utf-8') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
        
        # 최근 로그 조회
        response = await get_recent_logs()
        
        assert "logs" in response
        assert "total_count" in response
        # 유효한 JSON만 파싱됨
        assert len(response["logs"]) == 2
        assert response["total_count"] == 3

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.settings')
    async def test_get_recent_logs_exception_handling(self, mock_settings):
        """로그 조회 중 예외 처리 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "test_log.json"
        
        # 파일이 존재하지만 읽기 실패하는 경우 모킹
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                # 로그 조회 시도 (예외 발생)
                with pytest.raises(HTTPException) as exc_info:
                    await get_recent_logs()
                
                assert exc_info.value.status_code == 500
                assert "로그 조회 실패" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.settings')
    async def test_get_recent_logs_limit_to_10(self, mock_settings):
        """로그 10개 제한 테스트"""
        mock_settings.log_directory = self.test_log_dir
        mock_settings.log_filename = "test_log.json"
        
        # 15개의 테스트 로그 생성
        test_logs = []
        for i in range(15):
            test_logs.append({
                "timestamp": f"2024-01-01T00:{i:02d}:00",
                "service": f"test{i}",
                "status": "normal"
            })
        
        with open(self.test_log_file, 'w', encoding='utf-8') as f:
            for log in test_logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')
        
        # 최근 로그 조회
        response = await get_recent_logs()
        
        assert len(response["logs"]) == 10
        assert response["total_count"] == 15
        
        # 최근 10개만 반환되는지 확인 (마지막 10개)
        assert response["logs"][0]["service"] == "test5"
        assert response["logs"][-1]["service"] == "test14"

    @pytest.mark.asyncio
    @patch('app.routers.simulator_router.simulator_scheduler')
    @patch('app.routers.simulator_router.painting_surface_model_client')
    async def test_get_simulator_status_with_health_check_failure(self, mock_model_client, mock_scheduler):
        """헬스 체크 실패가 포함된 상태 조회 테스트"""
        # Mock 스케줄러 상태
        mock_scheduler.is_running = True
        mock_scheduler.get_status.return_value = {
            "running": True,
            "started_at": "2024-01-01T00:00:00",
            "jobs": ["data_simulation"]
        }
        
        # Mock 모델 클라이언트 헬스 체크 실패
        mock_model_client.health_check = AsyncMock(return_value=False)
        
        # 상태 조회
        response = await get_simulator_status()
        
        assert response["running"] is True
        assert response["painting_surface_service_health"] is False
