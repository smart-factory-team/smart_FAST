import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from app.services.azure_storage import AzureStorageService


class TestAzureStorageService:
    """Test suite for AzureStorageService class using pytest framework."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings fixture for testing."""
        with patch('app.services.azure_storage.settings') as mock_settings:
            mock_settings.azure_connection_string = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test123;EndpointSuffix=core.windows.net"
            mock_settings.azure_container_name = "test-container"
            mock_settings.painting_data_folder = "painting-process-equipment"
            yield mock_settings

    @pytest.fixture
    def azure_service(self, mock_settings):
        """Create AzureStorageService instance for testing."""
        return AzureStorageService()

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for testing."""
        data = {
            'machineId': ['MACHINE001', 'MACHINE002', 'MACHINE003'],
            'timeStamp': ['2023-01-01T10:00:00Z', '2023-01-01T10:01:00Z', '2023-01-01T10:02:00Z'],
            'Thick': [1.5, 2.0, 1.8],
            'PT_jo_V_1': [220.5, 221.0, 219.8],
            'PT_jo_A_Main_1': [15.2, 16.1, 14.9],
            'PT_jo_TP': [25.3, 26.1, 24.8]
        }
        return pd.DataFrame(data)

    class TestInitialization:
        """Test AzureStorageService initialization."""

        def test_initialization_with_settings(self, mock_settings):
            """Test successful initialization with proper settings."""
            service = AzureStorageService()
            
            assert service.connection_string == mock_settings.azure_connection_string
            assert service.container_name == mock_settings.azure_container_name
            assert service.current_index == 0
            assert service.cached_df is None

        def test_initialization_attributes(self, azure_service, mock_settings):
            """Test all initialization attributes are set correctly."""
            assert hasattr(azure_service, 'connection_string')
            assert hasattr(azure_service, 'container_name')
            assert hasattr(azure_service, 'current_index')
            assert hasattr(azure_service, 'cached_df')
            assert azure_service.current_index == 0
            assert azure_service.cached_df is None

    class TestListDataFiles:
        """Test list_data_files method."""

        @pytest.mark.asyncio
        async def test_list_data_files_success(self, azure_service, mock_settings):
            """Test successful listing of CSV files."""
            mock_blob1 = MagicMock()
            mock_blob1.name = "painting-process-equipment/data1.csv"
            mock_blob2 = MagicMock()
            mock_blob2.name = "painting-process-equipment/data2.csv"
            mock_blob3 = MagicMock()
            mock_blob3.name = "painting-process-equipment/config.txt"  # Should be filtered out

            mock_client = MagicMock()
            mock_container_client = AsyncMock()
            mock_client.get_container_client.return_value = mock_container_client
            
            async def async_iterator(items):
                for item in items:
                    yield item
            
            mock_container_client.list_blobs.return_value = async_iterator([
                mock_blob1, mock_blob2, mock_blob3
            ])

            with patch('app.services.azure_storage.BlobServiceClient.from_connection_string') as mock_from_conn_str:
                mock_from_conn_str.return_value.__aenter__.return_value = mock_client

                result = await azure_service.list_data_files()

                assert len(result) == 2
                assert "painting-process-equipment/data1.csv" in result
                assert "painting-process-equipment/data2.csv" in result
                assert "painting-process-equipment/config.txt" not in result
                assert result == sorted(result)  # Should be sorted

        @pytest.mark.asyncio
        async def test_list_data_files_no_connection_string(self, mock_settings):
            """Test list_data_files with no connection string."""
            mock_settings.azure_connection_string = None
            service = AzureStorageService()
            
            with pytest.raises(ValueError, match="Azure connection string이 설정되지 않았습니다."):
                await service.list_data_files()

        @pytest.mark.asyncio
        async def test_list_data_files_empty_connection_string(self, mock_settings):
            """Test list_data_files with empty connection string."""
            mock_settings.azure_connection_string = ""
            service = AzureStorageService()
            
            with pytest.raises(ValueError, match="Azure connection string이 설정되지 않았습니다."):
                await service.list_data_files()

        @pytest.mark.asyncio
        async def test_list_data_files_exception_handling(self, azure_service, capsys):
            """Test exception handling in list_data_files."""
            with patch('app.services.azure_storage.BlobServiceClient') as mock_client_class:
                mock_client_class.from_connection_string.side_effect = Exception("Connection failed")

                result = await azure_service.list_data_files()

                assert result == []
                captured = capsys.readouterr()
                assert "❌ Painting 데이터 파일 목록 조회 실패: Connection failed" in captured.out

        @pytest.mark.asyncio
        async def test_list_data_files_no_csv_files(self, azure_service, mock_settings):
            """Test list_data_files when no CSV files are found."""
            mock_blob1 = MagicMock()
            mock_blob1.name = "painting-process-equipment/data1.txt"
            mock_blob2 = MagicMock()
            mock_blob2.name = "painting-process-equipment/data2.json"

            mock_client = MagicMock()
            mock_container_client = AsyncMock()
            mock_client.get_container_client.return_value = mock_container_client
            
            async def async_iterator(items):
                for item in items:
                    yield item
            
            mock_container_client.list_blobs.return_value = async_iterator([
                mock_blob1, mock_blob2
            ])

            with patch('app.services.azure_storage.BlobServiceClient.from_connection_string') as mock_from_conn_str:
                mock_from_conn_str.return_value.__aenter__.return_value = mock_client

                result = await azure_service.list_data_files()

                assert result == []

        @pytest.mark.asyncio
        async def test_list_data_files_prefix_filtering(self, azure_service, mock_settings):
            """Test that only files with correct prefix are returned."""
            mock_blob1 = MagicMock()
            mock_blob1.name = "painting-process-equipment/data1.csv"
            mock_blob2 = MagicMock()
            mock_blob2.name = "other-folder/data2.csv"  # Should be filtered out

            mock_client = MagicMock()
            mock_container_client = AsyncMock()
            mock_client.get_container_client.return_value = mock_container_client
            
            async def async_iterator(items):
                for item in items:
                    yield item
            
            mock_container_client.list_blobs.return_value = async_iterator([
                mock_blob1, mock_blob2
            ])

            with patch('app.services.azure_storage.BlobServiceClient.from_connection_string') as mock_from_conn_str:
                mock_from_conn_str.return_value.__aenter__.return_value = mock_client

                result = await azure_service.list_data_files()

                # Verify prefix was used correctly
                mock_container_client.list_blobs.assert_called_once_with(
                    name_starts_with=f"{mock_settings.painting_data_folder}/"
                )
                assert len(result) == 1
                assert "painting-process-equipment/data1.csv" in result

    class TestReadCsvData:
        """Test read_csv_data method."""

        @pytest.mark.asyncio
        async def test_read_csv_data_success(self, azure_service, sample_csv_data, capsys):
            """Test successful CSV data reading."""
            csv_content = sample_csv_data.to_csv(index=False)
            
            mock_client = MagicMock()
            mock_blob_client = AsyncMock()
            mock_client.get_blob_client.return_value = mock_blob_client
            
            mock_blob_data = AsyncMock()
            mock_blob_client.download_blob.return_value = mock_blob_data
            mock_blob_data.readall.return_value = csv_content.encode('utf-8')

            with patch('app.services.azure_storage.BlobServiceClient.from_connection_string') as mock_from_conn_str:
                mock_from_conn_str.return_value.__aenter__.return_value = mock_client

                result = await azure_service.read_csv_data("test.csv")

                assert result is not None
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 3
                assert list(result.columns) == list(sample_csv_data.columns)
                
                captured = capsys.readouterr()
                assert "📁 파일 읽기 성공: test.csv (3 rows)" in captured.out

        @pytest.mark.asyncio
        async def test_read_csv_data_no_connection_string(self, mock_settings):
            """Test read_csv_data with no connection string."""
            mock_settings.azure_connection_string = None
            service = AzureStorageService()
            
            with pytest.raises(ValueError, match="Azure connection string이 설정되지 않았습니다."):
                await service.read_csv_data("test.csv")

        @pytest.mark.asyncio
        async def test_read_csv_data_empty_connection_string(self, mock_settings):
            """Test read_csv_data with empty connection string."""
            mock_settings.azure_connection_string = ""
            service = AzureStorageService()
            
            with pytest.raises(ValueError, match="Azure connection string이 설정되지 않았습니다."):
                await service.read_csv_data("test.csv")

        @pytest.mark.asyncio
        async def test_read_csv_data_exception_handling(self, azure_service, capsys):
            """Test exception handling in read_csv_data."""
            with patch('app.services.azure_storage.BlobServiceClient') as mock_client_class:
                mock_client_class.from_connection_string.side_effect = Exception("Download failed")

                result = await azure_service.read_csv_data("test.csv")

                assert result is None
                captured = capsys.readouterr()
                assert "❌ 파일 읽기 실패 (test.csv): Download failed" in captured.out

        @pytest.mark.asyncio
        async def test_read_csv_data_invalid_csv(self, azure_service, capsys):
            """Test reading invalid CSV data."""
            invalid_csv = "invalid,csv,content\nwith,broken\nlines"
            
            with patch('app.services.azure_storage.BlobServiceClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.from_connection_string.return_value = mock_client
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                
                mock_blob_client = AsyncMock()
                mock_client.get_blob_client.return_value = mock_blob_client
                
                mock_blob_data = AsyncMock()
                mock_blob_client.download_blob.return_value = mock_blob_data
                mock_blob_data.readall.return_value = invalid_csv.encode('utf-8')

                result = await azure_service.read_csv_data("invalid.csv")

                assert result is None
                captured = capsys.readouterr()
                assert "❌ 파일 읽기 실패 (invalid.csv):" in captured.out

        @pytest.mark.asyncio
        async def test_read_csv_data_blob_client_calls(self, azure_service, sample_csv_data):
            """Test that blob client is called with correct parameters."""
            csv_content = sample_csv_data.to_csv(index=False)
            
            mock_client = MagicMock()
            mock_blob_client = AsyncMock()
            mock_client.get_blob_client.return_value = mock_blob_client
            
            mock_blob_data = AsyncMock()
            mock_blob_client.download_blob.return_value = mock_blob_data
            mock_blob_data.readall.return_value = csv_content.encode('utf-8')

            with patch('app.services.azure_storage.BlobServiceClient.from_connection_string') as mock_from_conn_str:
                mock_from_conn_str.return_value.__aenter__.return_value = mock_client

                await azure_service.read_csv_data("test.csv")

                mock_client.get_blob_client.assert_called_once_with(
                    container=azure_service.container_name,
                    blob="test.csv"
                )

    class TestSimulateRealTimeData:
        """Test simulate_real_time_data method."""

        @pytest.mark.asyncio
        async def test_simulate_real_time_data_first_call(self, azure_service, sample_csv_data, capsys):
            """Test first call to simulate_real_time_data loads DataFrame."""
            with patch.object(azure_service, '_load_dataframe', new_callable=AsyncMock) as mock_load:
                azure_service.cached_df = sample_csv_data.copy()

                result = await azure_service.simulate_real_time_data()

                mock_load.assert_called_once()
                assert result is not None
                assert result['machineId'] == 'PAINT-MACHINE001'
                assert result['thick'] == 1.5
                assert azure_service.current_index == 1

        @pytest.mark.asyncio
        async def test_simulate_real_time_data_subsequent_calls(self, azure_service, sample_csv_data, capsys):
            """Test subsequent calls use cached DataFrame."""
            azure_service.cached_df = sample_csv_data.copy()
            azure_service.current_index = 1

            with patch.object(azure_service, '_load_dataframe', new_callable=AsyncMock) as mock_load:
                result = await azure_service.simulate_real_time_data()

                mock_load.assert_not_called()
                assert result is not None
                assert result['machineId'] == 'PAINT-MACHINE002'
                assert result['thick'] == 2.0
                assert azure_service.current_index == 2

        @pytest.mark.asyncio
        async def test_simulate_real_time_data_index_wraparound(self, azure_service, sample_csv_data):
            """Test index wraps around to 0 when reaching end of DataFrame."""
            azure_service.cached_df = sample_csv_data.copy()
            azure_service.current_index = 2  # Last index

            result = await azure_service.simulate_real_time_data()

            assert result is not None
            assert result['machineId'] == 'PAINT-MACHINE003'
            assert azure_service.current_index == 0  # Wrapped around

        @pytest.mark.asyncio
        async def test_simulate_real_time_data_no_cached_df(self, azure_service, capsys):
            """Test behavior when cached_df remains None after loading."""
            with patch.object(azure_service, '_load_dataframe', new_callable=AsyncMock):
                azure_service.cached_df = None

                result = await azure_service.simulate_real_time_data()

                assert result is None

        @pytest.mark.asyncio
        async def test_simulate_real_time_data_missing_columns(self, azure_service, capsys):
            """Test behavior with missing required columns."""
            incomplete_data = pd.DataFrame({
                'machineId': ['MACHINE001'],
                'timeStamp': ['2023-01-01T10:00:00Z'],
                'Thick': [1.5],
                # Missing PT_jo_V_1, PT_jo_A_Main_1, PT_jo_TP
            })
            azure_service.cached_df = incomplete_data

            result = await azure_service.simulate_real_time_data()

            assert result is None
            captured = capsys.readouterr()
            assert "❌ Painting 데이터 시뮬레이션 실패:" in captured.out

        @pytest.mark.asyncio
        async def test_simulate_real_time_data_data_structure(self, azure_service, sample_csv_data):
            """Test the structure of returned simulated data."""
            azure_service.cached_df = sample_csv_data.copy()
            azure_service.current_index = 0

            result = await azure_service.simulate_real_time_data()

            assert isinstance(result, dict)
            expected_keys = ['machineId', 'timeStamp', 'thick', 'voltage', 'current', 'temper', 'issue', 'isSolved']
            assert all(key in result for key in expected_keys)
            
            assert result['machineId'] == 'PAINT-MACHINE001'
            assert result['timeStamp'] == '2023-01-01T10:00:00Z'
            assert result['thick'] == 1.5
            assert result['voltage'] == 220.5
            assert result['current'] == 15.2
            assert result['temper'] == 25.3
            assert result['issue'] == ""
            assert result['isSolved'] is False

        @pytest.mark.asyncio
        async def test_simulate_real_time_data_type_conversion(self, azure_service):
            """Test proper type conversion of numeric values."""
            test_data = pd.DataFrame({
                'machineId': ['TEST001'],
                'timeStamp': ['2023-01-01T10:00:00Z'],
                'Thick': ['1.5'],  # String that should convert to float
                'PT_jo_V_1': ['220.5'],
                'PT_jo_A_Main_1': ['15.2'],
                'PT_jo_TP': ['25.3']
            })
            azure_service.cached_df = test_data

            result = await azure_service.simulate_real_time_data()

            assert isinstance(result['thick'], float)
            assert isinstance(result['voltage'], float)
            assert isinstance(result['current'], float)
            assert isinstance(result['temper'], float)

        @pytest.mark.asyncio
        async def test_simulate_real_time_data_exception_handling(self, azure_service, capsys):
            """Test exception handling in simulate_real_time_data."""
            with patch.object(azure_service, '_load_dataframe', side_effect=Exception("Load failed")):
                result = await azure_service.simulate_real_time_data()

                assert result is None
                captured = capsys.readouterr()
                assert "❌ Painting 데이터 시뮬레이션 실패: Load failed" in captured.out

        @pytest.mark.asyncio
        async def test_simulate_real_time_data_empty_dataframe(self, azure_service):
            """Test behavior with empty DataFrame."""
            azure_service.cached_df = pd.DataFrame()

            result = await azure_service.simulate_real_time_data()

            assert result is None

    class TestLoadDataframe:
        """Test _load_dataframe method."""

        @pytest.mark.asyncio
        async def test_load_dataframe_success(self, azure_service, sample_csv_data, capsys):
            """Test successful DataFrame loading."""
            mock_files = ["painting-process-equipment/data1.csv", "painting-process-equipment/data2.csv"]
            
            with patch.object(azure_service, 'list_data_files', return_value=mock_files), \
                 patch.object(azure_service, 'read_csv_data', return_value=sample_csv_data):

                await azure_service._load_dataframe()

                assert azure_service.cached_df is not None
                assert len(azure_service.cached_df) == 3
                assert azure_service.current_index == 0
                
                captured = capsys.readouterr()
                assert "✅ 데이터 캐싱 완료:" in captured.out

        @pytest.mark.asyncio
        async def test_load_dataframe_no_files(self, azure_service, capsys):
            """Test behavior when no files are found."""
            with patch.object(azure_service, 'list_data_files', return_value=[]):
                await azure_service._load_dataframe()

                assert azure_service.cached_df is None
                captured = capsys.readouterr()
                assert "⚠️ Painting 데이터 파일이 없습니다." in captured.out

        @pytest.mark.asyncio
        async def test_load_dataframe_read_failure(self, azure_service, capsys):
            """Test behavior when CSV reading fails."""
            mock_files = ["painting-process-equipment/data1.csv"]
            
            with patch.object(azure_service, 'list_data_files', return_value=mock_files), \
                 patch.object(azure_service, 'read_csv_data', return_value=None):

                await azure_service._load_dataframe()

                assert azure_service.cached_df is None
                assert azure_service.current_index == 0

        @pytest.mark.asyncio
        async def test_load_dataframe_uses_first_file(self, azure_service, sample_csv_data):
            """Test that _load_dataframe uses the first available file."""
            mock_files = ["painting-process-equipment/data1.csv", "painting-process-equipment/data2.csv"]
            
            with patch.object(azure_service, 'list_data_files', return_value=mock_files), \
                 patch.object(azure_service, 'read_csv_data', return_value=sample_csv_data) as mock_read:

                await azure_service._load_dataframe()

                mock_read.assert_called_once_with("painting-process-equipment/data1.csv")

        @pytest.mark.asyncio
        async def test_load_dataframe_exception_handling(self, azure_service, capsys):
            """Test exception handling in _load_dataframe."""
            with patch.object(azure_service, 'list_data_files', side_effect=Exception("List failed")):
                await azure_service._load_dataframe()

                captured = capsys.readouterr()
                assert "❌ DataFrame 로드 실패: List failed" in captured.out

    class TestEdgeCasesAndBoundaryConditions:
        """Test edge cases and boundary conditions."""

        @pytest.mark.asyncio
        async def test_single_row_dataframe(self, azure_service):
            """Test behavior with single row DataFrame."""
            single_row_data = pd.DataFrame({
                'machineId': ['MACHINE001'],
                'timeStamp': ['2023-01-01T10:00:00Z'],
                'Thick': [1.5],
                'PT_jo_V_1': [220.5],
                'PT_jo_A_Main_1': [15.2],
                'PT_jo_TP': [25.3]
            })
            azure_service.cached_df = single_row_data

            # First call
            result1 = await azure_service.simulate_real_time_data()
            assert result1 is not None
            assert azure_service.current_index == 0  # Should wrap back to 0

            # Second call should return same data
            result2 = await azure_service.simulate_real_time_data()
            assert result2 is not None
            assert result2['machineId'] == result1['machineId']

        @pytest.mark.asyncio
        async def test_very_large_dataframe_index_management(self, azure_service):
            """Test index management with large DataFrame."""
            large_data = pd.DataFrame({
                'machineId': [f'MACHINE{i:03d}' for i in range(1000)],
                'timeStamp': ['2023-01-01T10:00:00Z'] * 1000,
                'Thick': [1.5] * 1000,
                'PT_jo_V_1': [220.5] * 1000,
                'PT_jo_A_Main_1': [15.2] * 1000,
                'PT_jo_TP': [25.3] * 1000
            })
            azure_service.cached_df = large_data
            azure_service.current_index = 999

            result = await azure_service.simulate_real_time_data()

            assert result is not None
            assert azure_service.current_index == 0  # Should wrap around

        @pytest.mark.asyncio 
        async def test_nan_values_in_dataframe(self, azure_service):
            """Test handling of NaN values in DataFrame."""
            data_with_nan = pd.DataFrame({
                'machineId': ['MACHINE001'],
                'timeStamp': ['2023-01-01T10:00:00Z'],
                'Thick': [float('nan')],
                'PT_jo_V_1': [220.5],
                'PT_jo_A_Main_1': [15.2],
                'PT_jo_TP': [25.3]
            })
            azure_service.cached_df = data_with_nan

            result = await azure_service.simulate_real_time_data()

            assert result is not None
            assert pd.isna(result['thick'])

        @pytest.mark.asyncio
        async def test_unicode_and_special_characters(self, azure_service):
            """Test handling of unicode and special characters."""
            special_data = pd.DataFrame({
                'machineId': ['MACHINE_특수문자'],
                'timeStamp': ['2023-01-01T10:00:00Z'],
                'Thick': [1.5],
                'PT_jo_V_1': [220.5],
                'PT_jo_A_Main_1': [15.2],
                'PT_jo_TP': [25.3]
            })
            azure_service.cached_df = special_data

            result = await azure_service.simulate_real_time_data()

            assert result is not None
            assert result['machineId'] == 'PAINT-MACHINE_특수문자'

    class TestIntegrationScenarios:
        """Test integration-like scenarios that combine multiple methods."""

        @pytest.mark.asyncio
        async def test_full_workflow_simulation(self, azure_service, sample_csv_data):
            """Test complete workflow from listing files to data simulation."""
            mock_files = ["painting-process-equipment/data1.csv"]
            
            with patch.object(azure_service, 'list_data_files', return_value=mock_files), \
                 patch.object(azure_service, 'read_csv_data', return_value=sample_csv_data):

                # First simulate call should trigger loading
                result1 = await azure_service.simulate_real_time_data()
                assert result1 is not None
                assert result1['machineId'] == 'PAINT-MACHINE001'

                # Second call should use cached data
                result2 = await azure_service.simulate_real_time_data()
                assert result2 is not None
                assert result2['machineId'] == 'PAINT-MACHINE002'

                # Third call should use cached data
                result3 = await azure_service.simulate_real_time_data()
                assert result3 is not None
                assert result3['machineId'] == 'PAINT-MACHINE003'

                # Fourth call should wrap around
                result4 = await azure_service.simulate_real_time_data()
                assert result4 is not None
                assert result4['machineId'] == 'PAINT-MACHINE001'

        @pytest.mark.asyncio
        async def test_error_recovery_workflow(self, azure_service, sample_csv_data):
            """Test error recovery in complete workflow."""
            # First attempt fails
            with patch.object(azure_service, 'list_data_files', side_effect=Exception("Network error")):
                result1 = await azure_service.simulate_real_time_data()
                assert result1 is None

            # Second attempt succeeds
            mock_files = ["painting-process-equipment/data1.csv"]
            with patch.object(azure_service, 'list_data_files', return_value=mock_files), \
                 patch.object(azure_service, 'read_csv_data', return_value=sample_csv_data):
                
                result2 = await azure_service.simulate_real_time_data()
                assert result2 is not None
                assert result2['machineId'] == 'PAINT-MACHINE001'


class TestGlobalInstance:
    """Test the global azure_storage instance."""

    def test_global_instance_creation(self):
        """Test that global instance is created properly."""
        from app.services.azure_storage import azure_storage
        
        assert azure_storage is not None
        assert isinstance(azure_storage, AzureStorageService)
        assert hasattr(azure_storage, 'current_index')
        assert hasattr(azure_storage, 'cached_df')

    def test_global_instance_singleton_behavior(self):
        """Test singleton-like behavior of global instance."""
        from app.services.azure_storage import azure_storage
        
        # Modify the global instance
        original_index = azure_storage.current_index
        azure_storage.current_index = 999
        
        # Import again and verify it's the same instance
        from app.services.azure_storage import azure_storage as azure_storage2
        
        assert azure_storage2.current_index == 999
        
        # Reset for other tests
        azure_storage.current_index = original_index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])