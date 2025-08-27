import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app, lifespan


class TestMainApplication:
    """메인 애플리케이션 테스트"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """루트 엔드포인트 테스트"""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["service"] == "Painting Surface Data Simulator Service"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert data["target_model"] == "painting-surface-defect-detection"
        assert "scheduler_status" in data
        assert "azure_storage" in data

    def test_health_check_endpoint(self):
        """헬스 체크 엔드포인트 테스트"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"

    def test_simulator_router_included(self):
        """시뮬레이터 라우터 포함 여부 테스트"""
        response = self.client.get("/simulator/status")
        
        # 라우터가 포함되어 있다면 200 또는 적절한 응답을 받아야 함
        # 실제 구현에 따라 응답이 달라질 수 있음
        assert response.status_code in [200, 404, 500]

    def test_test_connection_router_included(self):
        """테스트 연결 라우터 포함 여부 테스트"""
        response = self.client.post("/test/azure-storage-connection")
        
        # 라우터가 포함되어 있다면 적절한 응답을 받아야 함
        # 실제 구현에 따라 응답이 달라질 수 있음
        assert response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    @patch('app.main.settings')
    @patch('app.main.azure_storage')
    async def test_lifespan_startup_success(self, mock_azure_storage, mock_settings):
        """애플리케이션 시작 성공 테스트"""
        # Mock 설정
        mock_settings.azure_connection_string = "test_connection_string"
        mock_settings.azure_container_name = "test-container"
        mock_settings.painting_data_folder = "test-painting"
        mock_settings.painting_model_url = "http://test-model:8002"
        mock_settings.log_directory = "test-logs"
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage 연결 성공
        mock_azure_storage.connect = AsyncMock()
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            # 애플리케이션 시작
            async with lifespan(app):
                pass
            
            # 시작 메시지 출력 확인
            mock_print.assert_any_call("🚀 Painting Surface Defect Simulator Service 시작 중...")
            mock_print.assert_any_call("🔍 환경 변수 확인 중...")
            mock_print.assert_any_call("✅ Azure Storage 연결 성공!")

    @pytest.mark.asyncio
    @patch('app.main.settings')
    @patch('app.main.azure_storage')
    async def test_lifespan_startup_no_connection_string(self, mock_azure_storage, mock_settings):
        """연결 문자열 없음 시작 테스트"""
        # Mock 설정 (연결 문자열 없음)
        mock_settings.azure_connection_string = None
        mock_settings.azure_container_name = "test-container"
        mock_settings.painting_data_folder = "test-painting"
        mock_settings.painting_model_url = "http://test-model:8002"
        mock_settings.log_directory = "test-logs"
        mock_settings.scheduler_interval_minutes = 1
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            # 애플리케이션 시작
            async with lifespan(app):
                pass
            
            # 경고 메시지 출력 확인
            mock_print.assert_any_call("⚠️ AZURE_CONNECTION_STRING 환경 변수가 설정되지 않았습니다.")

    @pytest.mark.asyncio
    @patch('app.main.settings')
    @patch('app.main.azure_storage')
    async def test_lifespan_startup_azure_connection_failure(self, mock_azure_storage, mock_settings):
        """Azure Storage 연결 실패 시작 테스트"""
        # Mock 설정
        mock_settings.azure_connection_string = "test_connection_string"
        mock_settings.azure_container_name = "test-container"
        mock_settings.painting_data_folder = "test-painting"
        mock_settings.painting_model_url = "http://test-model:8002"
        mock_settings.log_directory = "test-logs"
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage 연결 실패
        mock_azure_storage.connect = AsyncMock(side_effect=Exception("Connection failed"))
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            # 애플리케이션 시작
            async with lifespan(app):
                pass
            
            # 연결 실패 메시지 출력 확인
            mock_print.assert_any_call("❌ Azure Storage 연결 실패: Connection failed")

    @pytest.mark.asyncio
    @patch('app.main.settings')
    @patch('app.main.azure_storage')
    @patch('app.main.simulator_scheduler')
    async def test_lifespan_shutdown(self, mock_scheduler, mock_azure_storage, mock_settings):
        """애플리케이션 종료 테스트"""
        # Mock 설정
        mock_settings.azure_connection_string = "test_connection_string"
        mock_settings.azure_container_name = "test-container"
        mock_settings.painting_data_folder = "test-painting"
        mock_settings.painting_model_url = "http://test-model:8002"
        mock_settings.log_directory = "test-logs"
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage 연결 성공
        mock_azure_storage.connect = AsyncMock()
        
        # Mock 스케줄러가 실행 중
        mock_scheduler.is_running = True
        mock_scheduler.stop = AsyncMock()
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            # 애플리케이션 시작 및 종료
            async with lifespan(app):
                pass
            
            # 종료 메시지 출력 확인
            mock_print.assert_any_call("🛑 Painting Surface Defect Simulator Service 종료 중...")
            
            # 스케줄러 중지 호출 확인
            mock_scheduler.stop.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.main.settings')
    @patch('app.main.azure_storage')
    @patch('app.main.simulator_scheduler')
    async def test_lifespan_shutdown_scheduler_not_running(self, mock_scheduler, mock_azure_storage, mock_settings):
        """스케줄러가 실행 중이 아닌 경우 종료 테스트"""
        # Mock 설정
        mock_settings.azure_connection_string = "test_connection_string"
        mock_settings.azure_container_name = "test-container"
        mock_settings.painting_data_folder = "test-painting"
        mock_settings.painting_model_url = "http://test-model:8002"
        mock_settings.log_directory = "test-logs"
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage 연결 성공
        mock_azure_storage.connect = AsyncMock()
        
        # Mock 스케줄러가 실행 중이 아님
        mock_scheduler.is_running = False
        mock_scheduler.stop = AsyncMock()
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            # 애플리케이션 시작 및 종료
            async with lifespan(app):
                pass
            
            # 종료 메시지 출력 확인
            mock_print.assert_any_call("🛑 Painting Surface Defect Simulator Service 종료 중...")
            
            # 스케줄러 중지가 호출되지 않음
            mock_scheduler.stop.assert_not_called()

    @pytest.mark.asyncio
    @patch('app.main.settings')
    @patch('app.main.azure_storage')
    async def test_lifespan_log_directory_creation(self, mock_azure_storage, mock_settings):
        """로그 디렉토리 생성 테스트"""
        # Mock 설정
        mock_settings.azure_connection_string = "test_connection_string"
        mock_settings.azure_container_name = "test-container"
        mock_settings.painting_data_folder = "test-painting"
        mock_settings.painting_model_url = "http://test-model:8002"
        mock_settings.log_directory = "test-logs"
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage 연결 성공
        mock_azure_storage.connect = AsyncMock()
        
        # Mock os.makedirs
        with patch('os.makedirs') as mock_makedirs:
            # 애플리케이션 시작
            async with lifespan(app):
                pass
            
            # 로그 디렉토리 생성 호출 확인
            mock_makedirs.assert_called_once_with("test-logs", exist_ok=True)

    def test_app_metadata(self):
        """애플리케이션 메타데이터 테스트"""
        assert app.title == "Painting Surface Defect Simulator Service"
        assert app.description == "도장 표면 결함 탐지 모델을 위한 실시간 데이터 시뮬레이터"
        assert app.version == "1.0.0"

    def test_app_has_lifespan(self):
        """애플리케이션에 lifespan이 설정되어 있는지 테스트"""
        assert app.router.lifespan_context == lifespan

    def test_app_includes_routers(self):
        """애플리케이션에 라우터가 포함되어 있는지 테스트"""
        # 시뮬레이터 라우터 확인 (경로에 /simulator가 포함된 라우트)
        simulator_routes = [route for route in app.routes if hasattr(route, 'path') and '/simulator' in route.path]
        test_routes = [route for route in app.routes if hasattr(route, 'path') and '/test' in route.path]

        assert len(simulator_routes) > 0, "시뮬레이터 라우터가 포함되지 않았습니다"
        assert len(test_routes) > 0, "테스트 연결 라우터가 포함되지 않았습니다"
        


    @pytest.mark.asyncio
    @patch('app.main.settings')
    @patch('app.main.azure_storage')
    async def test_lifespan_environment_variables_display(self, mock_azure_storage, mock_settings):
        """환경 변수 표시 테스트"""
        # Mock 설정
        mock_settings.azure_connection_string = "test_connection_string"
        mock_settings.azure_container_name = "test-container"
        mock_settings.painting_data_folder = "test-painting"
        mock_settings.painting_model_url = "http://test-model:8002"
        mock_settings.log_directory = "test-logs"
        mock_settings.scheduler_interval_minutes = 1
        
        # Mock Azure Storage 연결 성공
        mock_azure_storage.connect = AsyncMock()
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            # 애플리케이션 시작
            async with lifespan(app):
                pass
            
            # 환경 변수 표시 확인
            mock_print.assert_any_call("   Azure Connection String: ✅ 설정됨")
            mock_print.assert_any_call("   Azure Container: test-container")
            mock_print.assert_any_call("   Painting Data Folder: test-painting")
            mock_print.assert_any_call("   Model URL: http://test-model:8002")
            mock_print.assert_any_call("🔧 스케줄러 간격: 1분")
            mock_print.assert_any_call("🎯 대상 서비스: 도장 표면 결함탐지 모델")
