import pytest
import pandas as pd
from unittest.mock import patch, AsyncMock
from app.services.azure_storage import AzureStorageService

@pytest.fixture
def sample_csv_data():
    """Provides a sample DataFrame for testing."""
    data = {
        'machineId': ['MACHINE001'],
        'timeStamp': ['2023-01-01T10:00:00Z'],
        'Thick': [1.5],
        'PT_jo_V_1': [220.5],
        'PT_jo_A_Main_1': [15.2],
        'PT_jo_TP': [25.3]
    }
    return pd.DataFrame(data)

@pytest.mark.asyncio
async def test_simulate_real_time_data(sample_csv_data):
    """Test the data simulation logic."""
    service = AzureStorageService()
    service.cached_df = sample_csv_data

    with patch.object(service, '_load_dataframe', new_callable=AsyncMock) as mock_load:
        result = await service.simulate_real_time_data()

        mock_load.assert_not_called() # Should use the cached dataframe

        assert result is not None
        assert result["machineId"] == "PAINT-MACHINE001"
        assert result["thick"] == 1.5
        assert isinstance(result["voltage"], float)

@pytest.mark.asyncio
async def test_load_dataframe():
    """Test the dataframe loading logic."""
    service = AzureStorageService()
    
    with patch.object(service, 'list_data_files', new_callable=AsyncMock) as mock_list, \
         patch.object(service, 'read_csv_data', new_callable=AsyncMock) as mock_read:
        
        mock_list.return_value = ['test.csv']
        mock_read.return_value = pd.DataFrame({'machineId': ['test']})

        await service._load_dataframe()

        mock_list.assert_awaited_once()
        mock_read.assert_awaited_once_with('test.csv')
        assert service.cached_df is not None
