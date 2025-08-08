import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import HTTPException, status, UploadFile
from datetime import datetime
from io import BytesIO

from app.routers.predict_router import get_prediction_service
from app.schemas import (
    PredictRequest,
    PredictResponse,
    FilePredictResponse, 
    BatchPredictResponse,
    PredictData,
    FilePredictData,
    BatchPredictData,
    BatchSummary
)
from app.core.exceptions import (
    InvalidImageException,
    ImageSizeException,
    UnsupportedImageFormatException
)
from app.services.prediction_service import PredictionService

# Testing Framework: pytest with asyncio support and mocking capabilities


class TestPredictRouter:
    """Test suite for the prediction router endpoints"""

    @pytest.fixture
    def mock_model_manager(self):
        """Mock model manager dependency"""
        return Mock()

    @pytest.fixture
    def mock_prediction_service(self):
        """Mock prediction service"""
        return AsyncMock(spec=PredictionService)

    @pytest.fixture
    def sample_predict_request(self):
        """Sample prediction request data"""
        return PredictRequest(
            image_url="https://example.com/test-image.jpg",
            metadata={"source": "test"}
        )

    @pytest.fixture
    def sample_predict_response(self):
        """Sample prediction response"""
        return PredictResponse(
            success=True,
            message="예측이 완료되었습니다",
            data=PredictData(
                predicted_label="defect",
                confidence=0.95,
                probabilities={"normal": 0.05, "defect": 0.95},
                processing_time=0.5
            ),
            request_id="test-request-id",
            timestamp=datetime.now()
        )

    @pytest.fixture
    def sample_upload_file(self):
        """Sample upload file for testing"""
        file_content = b"fake image content"
        file = UploadFile(
            filename="test_image.jpg",
            file=BytesIO(file_content),
            size=len(file_content),
            headers={"content-type": "image/jpeg"}
        )
        file.content_type = "image/jpeg"
        return file

    @pytest.fixture
    def sample_file_predict_response(self):
        """Sample file prediction response"""
        return FilePredictResponse(
            success=True,
            message="파일 예측이 완료되었습니다",
            data=FilePredictData(
                predicted_label="normal",
                confidence=0.85,
                probabilities={"normal": 0.85, "defect": 0.15},
                processing_time=0.3,
                file_info={
                    "filename": "test_image.jpg",
                    "size": 1024,
                    "content_type": "image/jpeg"
                }
            ),
            request_id="test-request-id",
            timestamp=datetime.now()
        )

    @pytest.fixture
    def sample_batch_response(self):
        """Sample batch prediction response"""
        return BatchPredictResponse(
            success=True,
            message="배치 예측이 완료되었습니다",
            data=BatchPredictData(
                results=[],
                summary=BatchSummary(
                    total_count=2,
                    successful_count=2,
                    failed_count=0,
                    processing_time=1.0
                )
            ),
            request_id="test-request-id",
            timestamp=datetime.now()
        )


class TestGetPredictionService:
    """Test the dependency injection for prediction service"""

    def test_get_prediction_service_returns_instance(self):
        """Test that get_prediction_service returns a PredictionService instance"""
        mock_model_manager = Mock()
        
        result = get_prediction_service(mock_model_manager)
        
        assert isinstance(result, PredictionService)
        assert result.model_manager == mock_model_manager


class TestPredictEndpoint:
    """Test cases for the /predict endpoint"""

    @pytest.mark.asyncio
    async def test_predict_success(self, mock_prediction_service, sample_predict_request, sample_predict_response):
        """Test successful prediction with valid request"""
        # Arrange
        mock_prediction_service.predict_from_data.return_value = sample_predict_response
        
        # Act
        with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
             patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
            from app.routers.predict_router import predict
            result = await predict(sample_predict_request, "test-request-id", mock_prediction_service)
        
        # Assert
        assert result.success is True
        assert result.request_id == "test-request-id"
        assert isinstance(result.timestamp, datetime)
        mock_prediction_service.predict_from_data.assert_called_once_with(sample_predict_request)

    @pytest.mark.asyncio
    async def test_predict_invalid_image_exception(self, mock_prediction_service, sample_predict_request):
        """Test prediction with invalid image exception"""
        # Arrange
        mock_prediction_service.predict_from_data.side_effect = InvalidImageException("Invalid image format")
        
        # Act & Assert
        with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
             patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
            from app.routers.predict_router import predict
            with pytest.raises(HTTPException) as exc_info:
                await predict(sample_predict_request, "test-request-id", mock_prediction_service)
            
            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
            assert exc_info.value.detail["success"] is False
            assert exc_info.value.detail["request_id"] == "test-request-id"

    @pytest.mark.asyncio
    async def test_predict_image_size_exception(self, mock_prediction_service, sample_predict_request):
        """Test prediction with image size exception"""
        # Arrange
        mock_prediction_service.predict_from_data.side_effect = ImageSizeException(15000000, 10000000)
        
        # Act & Assert
        with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
             patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
            from app.routers.predict_router import predict
            with pytest.raises(HTTPException) as exc_info:
                await predict(sample_predict_request, "test-request-id", mock_prediction_service)
            
            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_predict_unsupported_format_exception(self, mock_prediction_service, sample_predict_request):
        """Test prediction with unsupported image format exception"""
        # Arrange
        mock_prediction_service.predict_from_data.side_effect = UnsupportedImageFormatException(
            "image/webp", ["image/jpeg", "image/png"]
        )
        
        # Act & Assert
        with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
             patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
            from app.routers.predict_router import predict
            with pytest.raises(HTTPException) as exc_info:
                await predict(sample_predict_request, "test-request-id", mock_prediction_service)
            
            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_predict_general_exception(self, mock_prediction_service, sample_predict_request):
        """Test prediction with general exception"""
        # Arrange
        mock_prediction_service.predict_from_data.side_effect = Exception("Database connection failed")
        
        # Act & Assert
        with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
             patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
            from app.routers.predict_router import predict
            with pytest.raises(HTTPException) as exc_info:
                await predict(sample_predict_request, "test-request-id", mock_prediction_service)
            
            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert exc_info.value.detail["success"] is False
            assert exc_info.value.detail["message"] == "예측 처리 중 오류가 발생했습니다"


class TestPredictFileEndpoint:
    """Test cases for the /predict/file endpoint"""

    @pytest.mark.asyncio
    async def test_predict_file_success(self, mock_prediction_service, sample_upload_file, sample_file_predict_response):
        """Test successful file prediction"""
        # Arrange
        mock_prediction_service.predict_from_file.return_value = sample_file_predict_response
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png"]
            mock_settings.MAX_IMAGE_SIZE = 10000000
            
            # Act
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_file
                result = await predict_file(sample_upload_file, "test-request-id", mock_prediction_service)
        
        # Assert
        assert result.success is True
        assert result.request_id == "test-request-id"
        mock_prediction_service.predict_from_file.assert_called_once_with(sample_upload_file)

    @pytest.mark.asyncio
    async def test_predict_file_unsupported_format(self, mock_prediction_service, sample_upload_file):
        """Test file prediction with unsupported file format"""
        # Arrange
        sample_upload_file.content_type = "image/webp"
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png"]
            
            # Act & Assert
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_file
                with pytest.raises(HTTPException) as exc_info:
                    await predict_file(sample_upload_file, "test-request-id", mock_prediction_service)
                
                assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_predict_file_size_exceeded(self, mock_prediction_service, sample_upload_file):
        """Test file prediction with file size exceeded"""
        # Arrange
        sample_upload_file.size = 15000000  # 15MB
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png"]
            mock_settings.MAX_IMAGE_SIZE = 10000000  # 10MB
            
            # Act & Assert
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_file
                with pytest.raises(HTTPException) as exc_info:
                    await predict_file(sample_upload_file, "test-request-id", mock_prediction_service)
                
                assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_predict_file_no_size_attribute(self, mock_prediction_service, sample_upload_file, sample_file_predict_response):
        """Test file prediction when file has no size attribute"""
        # Arrange
        sample_upload_file.size = None  # No size attribute
        mock_prediction_service.predict_from_file.return_value = sample_file_predict_response
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png"]
            mock_settings.MAX_IMAGE_SIZE = 10000000
            
            # Act
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_file
                result = await predict_file(sample_upload_file, "test-request-id", mock_prediction_service)
        
        # Assert - Should pass size check when size is None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_predict_file_service_exception(self, mock_prediction_service, sample_upload_file):
        """Test file prediction with service exception during prediction"""
        # Arrange
        mock_prediction_service.predict_from_file.side_effect = Exception("Model inference failed")
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png"] 
            mock_settings.MAX_IMAGE_SIZE = 10000000
            
            # Act & Assert
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_file
                with pytest.raises(HTTPException) as exc_info:
                    await predict_file(sample_upload_file, "test-request-id", mock_prediction_service)
                
                assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                assert exc_info.value.detail["message"] == "파일 예측 처리 중 오류가 발생했습니다"


class TestPredictBatchEndpoint:
    """Test cases for the /predict/batch endpoint"""

    @pytest.fixture
    def sample_files_list(self):
        """Create a list of sample upload files for batch testing"""
        files = []
        for i in range(3):
            file_content = f"fake image content {i}".encode()
            file = UploadFile(
                filename=f"test_image_{i}.jpg",
                file=BytesIO(file_content),
                size=len(file_content),
                headers={"content-type": "image/jpeg"}
            )
            file.content_type = "image/jpeg"
            files.append(file)
        return files

    @pytest.mark.asyncio
    async def test_predict_batch_success(self, mock_prediction_service, sample_files_list, sample_batch_response):
        """Test successful batch prediction"""
        # Arrange
        mock_prediction_service.predict_batch.return_value = sample_batch_response
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.BATCH_SIZE_LIMIT = 10
            
            # Act
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_batch
                result = await predict_batch(sample_files_list, "test-request-id", mock_prediction_service)
        
        # Assert
        assert result.success is True
        assert result.request_id == "test-request-id"
        mock_prediction_service.predict_batch.assert_called_once_with(sample_files_list)

    @pytest.mark.asyncio
    async def test_predict_batch_size_limit_exceeded(self, mock_prediction_service):
        """Test batch prediction with size limit exceeded"""
        # Arrange
        large_files_list = [Mock() for _ in range(15)]  # 15 files
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.BATCH_SIZE_LIMIT = 10
            
            # Act & Assert
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_batch
                with pytest.raises(HTTPException) as exc_info:
                    await predict_batch(large_files_list, "test-request-id", mock_prediction_service)
                
                assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
                assert "배치 크기 초과" in exc_info.value.detail["message"]
                assert exc_info.value.detail["current_count"] == 15
                assert exc_info.value.detail["max_count"] == 10

    @pytest.mark.asyncio
    async def test_predict_batch_empty_files_list(self, mock_prediction_service, sample_batch_response):
        """Test batch prediction with empty files list"""
        # Arrange
        empty_files_list = []
        mock_prediction_service.predict_batch.return_value = sample_batch_response
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.BATCH_SIZE_LIMIT = 10
            
            # Act
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_batch
                result = await predict_batch(empty_files_list, "test-request-id", mock_prediction_service)
        
        # Assert
        assert result.success is True
        mock_prediction_service.predict_batch.assert_called_once_with(empty_files_list)

    @pytest.mark.asyncio
    async def test_predict_batch_service_exception(self, mock_prediction_service, sample_files_list):
        """Test batch prediction with service exception"""
        # Arrange
        mock_prediction_service.predict_batch.side_effect = Exception("Batch processing failed")
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.BATCH_SIZE_LIMIT = 10
            
            # Act & Assert
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_batch
                with pytest.raises(HTTPException) as exc_info:
                    await predict_batch(sample_files_list, "test-request-id", mock_prediction_service)
                
                assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                assert exc_info.value.detail["message"] == "배치 예측 처리 중 오류가 발생했습니다"

    @pytest.mark.asyncio
    async def test_predict_batch_http_exception_passthrough(self, mock_prediction_service, sample_files_list):
        """Test that HTTPExceptions are passed through without modification"""
        # Arrange
        original_http_exception = HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
        mock_prediction_service.predict_batch.side_effect = original_http_exception
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.BATCH_SIZE_LIMIT = 10
            
            # Act & Assert
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_batch
                with pytest.raises(HTTPException) as exc_info:
                    await predict_batch(sample_files_list, "test-request-id", mock_prediction_service)
                
                # Should be the same HTTPException that was raised
                assert exc_info.value == original_http_exception


class TestLoggingBehavior:
    """Test logging behavior across all endpoints"""

    @pytest.mark.asyncio
    async def test_predict_logging_on_success(self, mock_prediction_service, sample_predict_request, sample_predict_response):
        """Test that successful predictions are logged correctly"""
        mock_prediction_service.predict_from_data.return_value = sample_predict_response
        
        with patch('app.routers.predict_router.logger') as mock_logger, \
             patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
             patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
            from app.routers.predict_router import predict
            await predict(sample_predict_request, "test-request-id", mock_prediction_service)
        
        # Assert logging calls
        assert mock_logger.info.call_count >= 2  # Start and completion logs
        mock_logger.info.assert_any_call("예측 요청 시작 - Request ID: test-request-id")

    @pytest.mark.asyncio
    async def test_predict_logging_on_validation_error(self, mock_prediction_service, sample_predict_request):
        """Test that validation errors are logged as warnings"""
        mock_prediction_service.predict_from_data.side_effect = InvalidImageException("Invalid image")
        
        with patch('app.routers.predict_router.logger') as mock_logger, \
             patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
             patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
            from app.routers.predict_router import predict
            with pytest.raises(HTTPException):
                await predict(sample_predict_request, "test-request-id", mock_prediction_service)
        
        # Assert warning log
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio 
    async def test_predict_logging_on_general_error(self, mock_prediction_service, sample_predict_request):
        """Test that general errors are logged with exc_info"""
        mock_prediction_service.predict_from_data.side_effect = Exception("Unexpected error")
        
        with patch('app.routers.predict_router.logger') as mock_logger, \
             patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
             patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
            from app.routers.predict_router import predict
            with pytest.raises(HTTPException):
                await predict(sample_predict_request, "test-request-id", mock_prediction_service)
        
        # Assert error log with exc_info
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert call_args[1]['exc_info'] is True


class TestTimestampBehavior:
    """Test timestamp handling across all endpoints"""

    @pytest.mark.asyncio
    async def test_predict_sets_current_timestamp(self, mock_prediction_service, sample_predict_request, sample_predict_response):
        """Test that predict endpoint sets current timestamp"""
        mock_prediction_service.predict_from_data.return_value = sample_predict_response
        
        with patch('app.routers.predict_router.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict
                result = await predict(sample_predict_request, "test-request-id", mock_prediction_service)
        
        assert result.timestamp == mock_now

    @pytest.mark.asyncio
    async def test_error_response_includes_timestamp(self, mock_prediction_service, sample_predict_request):
        """Test that error responses include current timestamp"""
        mock_prediction_service.predict_from_data.side_effect = Exception("Test error")
        
        with patch('app.routers.predict_router.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict
                with pytest.raises(HTTPException) as exc_info:
                    await predict(sample_predict_request, "test-request-id", mock_prediction_service)
        
        assert exc_info.value.detail["timestamp"] == mock_now.isoformat()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_predict_with_none_request_id(self, mock_prediction_service, sample_predict_request, sample_predict_response):
        """Test prediction with None request_id"""
        mock_prediction_service.predict_from_data.return_value = sample_predict_response
        
        with patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
            from app.routers.predict_router import predict
            result = await predict(sample_predict_request, None, mock_prediction_service)
        
        assert result.request_id is None

    @pytest.mark.asyncio
    async def test_predict_file_with_empty_filename(self, mock_prediction_service, sample_file_predict_response):
        """Test file prediction with empty filename"""
        file_content = b"fake image content"
        file = UploadFile(
            filename="",  # Empty filename
            file=BytesIO(file_content),
            size=len(file_content),
            headers={"content-type": "image/jpeg"}
        )
        file.content_type = "image/jpeg"
        
        mock_prediction_service.predict_from_file.return_value = sample_file_predict_response
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png"]
            mock_settings.MAX_IMAGE_SIZE = 10000000
            
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_file
                result = await predict_file(file, "test-request-id", mock_prediction_service)
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_batch_predict_exactly_at_limit(self, mock_prediction_service, sample_batch_response):
        """Test batch prediction with exactly the maximum number of files"""
        # Create exactly BATCH_SIZE_LIMIT files
        files_list = [Mock() for _ in range(5)]
        mock_prediction_service.predict_batch.return_value = sample_batch_response
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.BATCH_SIZE_LIMIT = 5  # Exactly 5 files
            
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_batch
                result = await predict_batch(files_list, "test-request-id", mock_prediction_service)
        
        # Should succeed without raising exception
        assert result.success is True

    @pytest.mark.asyncio
    async def test_predict_file_with_zero_size(self, mock_prediction_service, sample_file_predict_response):
        """Test file prediction with zero size file"""
        file = UploadFile(
            filename="empty.jpg",
            file=BytesIO(b""),
            size=0,  # Zero size
            headers={"content-type": "image/jpeg"}
        )
        file.content_type = "image/jpeg"
        
        mock_prediction_service.predict_from_file.return_value = sample_file_predict_response
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png"]
            mock_settings.MAX_IMAGE_SIZE = 10000000
            
            with patch('app.routers.predict_router.get_request_id', return_value="test-request-id"), \
                 patch('app.routers.predict_router.get_prediction_service', return_value=mock_prediction_service):
                from app.routers.predict_router import predict_file
                result = await predict_file(file, "test-request-id", mock_prediction_service)
        
        # Should pass size check (0 <= MAX_IMAGE_SIZE)
        assert result.success is True