import pytest
from unittest.mock import patch, AsyncMock
from app.services.backend_client import BackendClient

@pytest.mark.asyncio
async def test_send_to_backend():
    """Test that the backend client sends data correctly."""
    client = BackendClient()
    test_data = {"test_key": "test_value"}
    test_url = "http://test-backend.com/api"

    with patch('httpx.AsyncClient') as mock_async_client:
        mock_client_instance = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance

        await client.send_to_backend(test_data, test_url)

        mock_client_instance.post.assert_awaited_once_with(test_url, json=test_data)
