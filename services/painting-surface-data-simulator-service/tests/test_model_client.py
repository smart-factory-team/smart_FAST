import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from app.services.model_client import PaintingSurfaceModelClient


class TestPaintingSurfaceModelClient:
    """도장 표면 모델 클라이언트 테스트"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        self.test_service_url = "http://test-model:8002"
        self.test_timeout = 5
        self.test_max_retries = 2

    @patch('app.services.model_client.settings')
    def test_model_client_initialization(self, mock_settings):
        """모델 클라이언트 초기화 테스트"""
        mock_settings.http_timeout = self.test_timeout
        mock_settings.max_retries = self.test_max_retries
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        assert client.timeout.connect == self.test_timeout
        assert client.max_retries == self.test_max_retries
        assert client.service_name == "painting-surface"
        assert client.service_url == self.test_service_url

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_predict_painting_surface_data_success(self, mock_settings):
        """도장 표면 데이터 예측 성공 테스트"""
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        # Mock Azure Storage 다운로드
        with patch.object(client, '_download_image_from_azure') as mock_download:
            mock_download.return_value = b"test_image_data"
            
            # Mock HTTP 요청
            with patch.object(client, '_predict_with_file_upload') as mock_predict:
                mock_predict.return_value = {
                    "predictions": [],
                    "status": "normal",
                    "defect_count": 0,
                    "total_count": 1
                }
                
                # Mock 상세 로깅
                with patch.object(client, '_log_detailed_prediction_result') as mock_log:
                    result = await client.predict_painting_surface_data(["test1.jpg", "test2.jpg"])
                    
                    assert result is not None
                    assert "images" in result
                    assert "combined" in result
                    assert len(result["images"]) == 2
                    assert result["combined"]["status"] == "normal"
                    assert result["combined"]["defect_count"] == 0
                    assert result["combined"]["total_count"] == 2

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_predict_painting_surface_data_no_service_url(self, mock_settings):
        """서비스 URL 없음 테스트"""
        mock_settings.model_service_url = None
        
        client = PaintingSurfaceModelClient()
        
        result = await client.predict_painting_surface_data(["test.jpg"])
        
        assert result is None

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_predict_painting_surface_data_download_failure(self, mock_settings):
        """이미지 다운로드 실패 테스트"""
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        # Mock Azure Storage 다운로드 실패
        with patch.object(client, '_download_image_from_azure') as mock_download:
            mock_download.return_value = None
            
            result = await client.predict_painting_surface_data(["test.jpg"])
            
            assert result is None

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_predict_painting_surface_data_prediction_failure(self, mock_settings):
        """예측 요청 실패 테스트"""
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        # Mock Azure Storage 다운로드 성공
        with patch.object(client, '_download_image_from_azure') as mock_download:
            mock_download.return_value = b"test_image_data"
            
            # Mock HTTP 요청 실패
            with patch.object(client, '_predict_with_file_upload') as mock_predict:
                mock_predict.return_value = None
                
                result = await client.predict_painting_surface_data(["test.jpg"])
                
                assert result is None

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_predict_painting_surface_data_exception_handling(self, mock_settings):
        """예외 처리 테스트"""
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        # Mock Azure Storage 다운로드에서 예외 발생
        with patch.object(client, '_download_image_from_azure') as mock_download:
            mock_download.side_effect = Exception("Download error")
            
            result = await client.predict_painting_surface_data(["test.jpg"])
            
            assert result is None

    def test_combine_painting_results_normal(self):
        """정상 결과 조합 테스트"""
        client = PaintingSurfaceModelClient()
        
        image_results = [
            {"predictions": [], "status": "normal"},
            {"predictions": [], "status": "normal"}
        ]
        
        combined = client._combine_painting_results(image_results)
        
        assert combined["status"] == "normal"
        assert combined["defect_count"] == 0
        assert combined["total_count"] == 2
        assert combined["defect_ratio"] == 0.0

    def test_combine_painting_results_anomaly(self):
        """결함 결과 조합 테스트"""
        client = PaintingSurfaceModelClient()
        
        image_results = [
            {"predictions": [{"defect": "scratch"}], "status": "anomaly"},
            {"predictions": [], "status": "normal"}
        ]
        
        combined = client._combine_painting_results(image_results)
        
        assert combined["status"] == "anomaly"
        assert combined["defect_count"] == 1
        assert combined["total_count"] == 2
        assert combined["defect_ratio"] == 0.5

    def test_combine_painting_results_empty(self):
        """빈 결과 조합 테스트"""
        client = PaintingSurfaceModelClient()
        
        combined = client._combine_painting_results([])
        
        assert combined["status"] == "error"
        assert "message" in combined

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_health_check_success(self, mock_settings):
        """헬스 체크 성공 테스트"""
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        # Mock HTTP 응답
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await client.health_check()
            
            assert result is True

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_health_check_failure(self, mock_settings):
        """헬스 체크 실패 테스트"""
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        # Mock HTTP 응답 실패
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await client.health_check()
            
            assert result is False

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_health_check_exception(self, mock_settings):
        """헬스 체크 예외 테스트"""
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        # Mock HTTP 요청에서 예외 발생
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = Exception("Connection error")
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await client.health_check()
            
            assert result is False

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_health_check_no_service_url(self, mock_settings):
        """서비스 URL 없음 헬스 체크 테스트"""
        mock_settings.model_service_url = None
        
        client = PaintingSurfaceModelClient()
        
        result = await client.health_check()
        
        assert result is False

    @patch('app.services.model_client.settings')
    def test_log_detailed_prediction_result(self, mock_settings):
        """상세 예측 결과 로깅 테스트"""
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        # 콘솔 출력 모킹
        with patch('builtins.print') as mock_print:
            image_file = "test.jpg"
            result = {
                "predictions": [{"defect": "scratch", "confidence": 0.95}],
                "status": "anomaly"
            }
            
            client._log_detailed_prediction_result(image_file, result)
            
            # 콘솔 출력 확인
            mock_print.assert_called()

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_download_image_from_azure(self, mock_settings):
        """Azure에서 이미지 다운로드 테스트"""
        mock_settings.model_service_url = self.test_service_url
        
        client = PaintingSurfaceModelClient()
        
        # Mock Azure Storage 서비스 (메서드 자체를 Mock)
        with patch.object(client, '_download_image_from_azure', new_callable=AsyncMock) as mock_download:
            mock_download.return_value = b"test_image_data"
            
            result = await client._download_image_from_azure("test.jpg")
            
            assert result == b"test_image_data"
            mock_download.assert_called_once_with("test.jpg")

    @pytest.mark.asyncio
    @patch('app.services.model_client.settings')
    async def test_predict_with_file_upload(self, mock_settings):
        """파일 업로드 방식 예측 요청 테스트"""
        mock_settings.model_service_url = self.test_service_url
        mock_settings.max_retries = 2  # max_retries를 실제 정수로 설정
        
        client = PaintingSurfaceModelClient()
        
        # Mock HTTP 요청
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "predictions": [],
                "status": "normal"
            }
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            image_data = b"test_image_data"
            image_file = "test.jpg"
            confidence_threshold = 0.5
            
            result = await client._predict_with_file_upload(
                "http://test-model:8002/predict/file",
                image_data,
                image_file,
                confidence_threshold
            )
            
            assert result is not None
            assert result["status"] == "normal"
            mock_client.post.assert_called_once()
