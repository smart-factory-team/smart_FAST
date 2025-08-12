import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from azure.core.exceptions import AzureError, ClientAuthenticationError
from app.services.azure_storage import AzureStorageService


class TestAzureStorageService:
    """Azure Storage 서비스 테스트"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        self.test_connection_string = "test_connection_string"
        self.test_container_name = "test-container"
        self.test_data_folder = "test-painting"

    @patch('app.services.azure_storage.settings')
    def test_azure_storage_initialization(self, mock_settings):
        """Azure Storage 서비스 초기화 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        mock_settings.azure_container_name = self.test_container_name
        mock_settings.painting_data_folder = self.test_data_folder
        
        storage_service = AzureStorageService()
        
        assert storage_service.connection_string == self.test_connection_string
        assert storage_service.container_name == self.test_container_name
        assert storage_service.client is None
        assert storage_service.image_index == 0

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_connect_success(self, mock_blob_service_client, mock_settings):
        """연결 성공 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        mock_settings.azure_container_name = self.test_container_name
        
        # Mock 클라이언트 설정
        mock_client = MagicMock()
        mock_container_client = AsyncMock()
        mock_properties = MagicMock()
        mock_properties.name = self.test_container_name
        mock_properties.created_on = "2024-01-01"
        
        mock_container_client.get_container_properties = AsyncMock(return_value=mock_properties)
        mock_client.get_container_client.return_value = mock_container_client
        mock_blob_service_client.from_connection_string.return_value = mock_client
        
        storage_service = AzureStorageService()
        
        # 연결 테스트
        await storage_service.connect()
        
        assert storage_service.client == mock_client
        mock_blob_service_client.from_connection_string.assert_called_once_with(
            self.test_connection_string
        )

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_connect_failure_no_connection_string(self, mock_blob_service_client, mock_settings):
        """연결 문자열 없음 실패 테스트"""
        mock_settings.azure_connection_string = None
        
        storage_service = AzureStorageService()
        
        with pytest.raises(ValueError, match="Azure connection string이 설정되지 않았습니다."):
            await storage_service.connect()

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_connect_failure_authentication_error(self, mock_blob_service_client, mock_settings):
        """인증 실패 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        
        # 인증 오류 시뮬레이션
        mock_blob_service_client.from_connection_string.side_effect = ClientAuthenticationError("Auth failed")
        
        storage_service = AzureStorageService()
        
        with pytest.raises(ClientAuthenticationError):
            await storage_service.connect()

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_connect_failure_general_error(self, mock_blob_service_client, mock_settings):
        """일반 오류 실패 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        
        # 일반 오류 시뮬레이션
        mock_blob_service_client.from_connection_string.side_effect = Exception("General error")
        
        storage_service = AzureStorageService()
        
        with pytest.raises(Exception, match="General error"):
            await storage_service.connect()

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    async def test_disconnect(self, mock_settings):
        """연결 종료 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        
        storage_service = AzureStorageService()
        storage_service.client = AsyncMock()
        
        await storage_service.disconnect()
        
        storage_service.client.close.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_list_data_files_success(self, mock_blob_service_client, mock_settings):
        """데이터 파일 목록 조회 성공 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        mock_settings.azure_container_name = self.test_container_name
        mock_settings.painting_data_folder = self.test_data_folder
        
        # Mock 클라이언트 설정
        mock_client = MagicMock()
        mock_container_client = AsyncMock()
        
        # Mock blob 객체들
        mock_blob1 = MagicMock()
        mock_blob1.name = f"{self.test_data_folder}/image1.jpg"
        mock_blob2 = MagicMock()
        mock_blob2.name = f"{self.test_data_folder}/image2.png"
        mock_blob3 = MagicMock()
        mock_blob3.name = f"{self.test_data_folder}/document.txt"  # 이미지가 아닌 파일
        
        # list_blobs는 비동기 이터레이터를 반환해야 함
        async def mock_list_blobs(*args, **kwargs):
            for blob in [mock_blob1, mock_blob2, mock_blob3]:
                yield blob
        mock_container_client.list_blobs = mock_list_blobs
        mock_client.get_container_client.return_value = mock_container_client
        mock_blob_service_client.from_connection_string.return_value = mock_client
        
        storage_service = AzureStorageService()
        
        # 파일 목록 조회
        files = await storage_service.list_data_files()
        
        expected_files = [f"{self.test_data_folder}/image1.jpg", f"{self.test_data_folder}/image2.png"]
        assert files == expected_files

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_list_data_files_authentication_error(self, mock_blob_service_client, mock_settings):
        """인증 오류로 인한 파일 목록 조회 실패 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        mock_settings.azure_container_name = self.test_container_name
        
        # Mock 클라이언트 설정
        mock_client = MagicMock()
        mock_container_client = MagicMock()
        mock_container_client.list_blobs.side_effect = ClientAuthenticationError("Auth failed")
        
        mock_client.get_container_client.return_value = mock_container_client
        mock_blob_service_client.from_connection_string.return_value = mock_client
        
        storage_service = AzureStorageService()
        
        # 파일 목록 조회 (인증 오류)
        files = await storage_service.list_data_files()
        
        assert files == []

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_list_data_files_general_error(self, mock_blob_service_client, mock_settings):
        """일반 오류로 인한 파일 목록 조회 실패 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        mock_settings.azure_container_name = self.test_container_name
        
        # Mock 클라이언트 설정
        mock_client = MagicMock()
        mock_container_client = MagicMock()
        mock_container_client.list_blobs.side_effect = Exception("General error")
        
        mock_client.get_container_client.return_value = mock_container_client
        mock_blob_service_client.from_connection_string.return_value = mock_client
        
        storage_service = AzureStorageService()
        
        # 파일 목록 조회 (일반 오류)
        files = await storage_service.list_data_files()
        
        assert files == []

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_read_image_data_success(self, mock_blob_service_client, mock_settings):
        """이미지 데이터 읽기 성공 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        mock_settings.azure_container_name = self.test_container_name
        
        # Mock 클라이언트 설정
        mock_client = MagicMock()
        mock_blob_client = AsyncMock()
        mock_blob_data = MagicMock()
        # readall은 Azure SDK에서 비동기 함수임
        mock_blob_data.readall = AsyncMock(return_value=b"test_image_data")
        
        mock_blob_client.download_blob = AsyncMock(return_value=mock_blob_data)
        mock_client.get_blob_client.return_value = mock_blob_client
        mock_blob_service_client.from_connection_string.return_value = mock_client
        
        storage_service = AzureStorageService()
        
        # Azure Storage 연결 Mock 설정
        storage_service.client = mock_client
        storage_service.image_index = 0  # image_index를 실제 정수로 설정
        
        # 이미지 데이터 읽기
        image_data = await storage_service.read_image_data("test.jpg")
        
        assert image_data == b"test_image_data"

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_read_image_data_failure(self, mock_blob_service_client, mock_settings):
        """이미지 데이터 읽기 실패 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        mock_settings.azure_container_name = self.test_container_name
        
        # Mock 클라이언트 설정
        mock_client = MagicMock()
        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.side_effect = Exception("Download failed")
        
        mock_client.get_blob_client.return_value = mock_blob_client
        mock_blob_service_client.from_connection_string.return_value = mock_client
        
        storage_service = AzureStorageService()
        
        # 이미지 데이터 읽기 (실패)
        image_data = await storage_service.read_image_data("test.jpg")
        
        assert image_data is None

    @pytest.mark.asyncio
    @patch('app.services.azure_storage.settings')
    @patch('app.services.azure_storage.BlobServiceClient')
    async def test_simulate_painting_surface_data(self, mock_blob_service_client, mock_settings):
        """도장 표면 데이터 시뮬레이션 테스트"""
        mock_settings.azure_connection_string = self.test_connection_string
        mock_settings.azure_container_name = self.test_container_name
        mock_settings.painting_data_folder = self.test_data_folder
        mock_settings.batch_size = 10  # batch_size를 실제 정수로 설정
        
        # Mock 클라이언트 설정
        mock_client = MagicMock()
        mock_container_client = AsyncMock()
        
        # Mock blob 객체들
        mock_blob1 = MagicMock()
        mock_blob1.name = f"{self.test_data_folder}/image1.jpg"
        mock_blob2 = MagicMock()
        mock_blob2.name = f"{self.test_data_folder}/image2.png"
        
        # list_blobs는 비동기 이터레이터를 반환해야 함
        async def mock_list_blobs(*args, **kwargs):
            for blob in [mock_blob1, mock_blob2]:
                yield blob
        mock_container_client.list_blobs = mock_list_blobs
        mock_client.get_container_client.return_value = mock_container_client
        mock_blob_service_client.from_connection_string.return_value = mock_client
        
        storage_service = AzureStorageService()
        
        # Azure Storage 연결 Mock 설정
        storage_service.client = mock_client
        storage_service.image_index = 0  # image_index를 실제 정수로 설정
        
        # 데이터 시뮬레이션
        simulated_data = await storage_service.simulate_painting_surface_data()
        
        assert "images" in simulated_data
        assert "total_images" in simulated_data
        assert "batch_size" in simulated_data
        assert len(simulated_data["images"]) == 2
        assert simulated_data["images"][0] == f"{self.test_data_folder}/image1.jpg"
        assert simulated_data["images"][1] == f"{self.test_data_folder}/image2.png"
