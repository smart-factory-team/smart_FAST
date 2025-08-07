import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock, call
import json

from app.services.model_client import ModelClient, model_client


class TestModelClient:
    """
    Comprehensive unit tests for ModelClient class using pytest and unittest.mock.
    Testing framework: pytest with pytest-asyncio for async testing.
    """

    @pytest.fixture
    def client(self):
        """Create a fresh ModelClient instance for each test."""
        return ModelClient()

    @pytest.fixture
    def mock_settings(self):
        """Mock settings configuration."""
        with patch('app.services.model_client.settings') as mock_settings:
            mock_settings.http_timeout = 30.0
            mock_settings.max_retries = 3
            mock_settings.model_services = {
                'welding-machine': 'http://localhost:8001',
                'test-service': 'http://localhost:8002'
            }
            yield mock_settings

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing predictions."""
        return {
            'current_data': {
                'values': [1.2, 1.3, 1.4, 1.5],
                'timestamp': '2023-01-01T00:00:00Z'
            },
            'vibration_data': {
                'values': [0.1, 0.2, 0.3, 0.4],
                'timestamp': '2023-01-01T00:00:00Z'
            }
        }

    # Test ModelClient initialization
    def test_init_with_default_settings(self, mock_settings):
        """Test ModelClient initialization with default settings."""
        client = ModelClient()
        assert client.timeout.connect == mock_settings.http_timeout
        assert client.timeout.read == mock_settings.http_timeout
        assert client.max_retries == mock_settings.max_retries

    # Tests for predict method - Happy Path
    @pytest.mark.asyncio
    async def test_predict_success(self, client, mock_settings):
        """Test successful prediction request."""
        service_name = 'test-service'
        data = {'input': 'test'}
        expected_response = {'status': 'normal', 'mae': 0.5}

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response)

            result = await client.predict(service_name, data)

            assert result == expected_response
            mock_client.return_value.__aenter__.return_value.post.assert_called_once_with(
                'http://localhost:8002/api/predict', json=data
            )

    @pytest.mark.asyncio
    async def test_predict_unknown_service(self, client, mock_settings):
        """Test prediction with unknown service name."""
        result = await client.predict('unknown-service', {'data': 'test'})
        assert result is None

    @pytest.mark.asyncio
    async def test_predict_http_error_with_retries(self, client, mock_settings):
        """Test prediction with HTTP error and retry logic."""
        service_name = 'test-service'
        data = {'input': 'test'}

        with patch('httpx.AsyncClient') as mock_client, \
                patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            mock_response = Mock()
            mock_response.status_code = 500

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response)

            result = await client.predict(service_name, data)

            assert result is None
            assert mock_client.return_value.__aenter__.return_value.post.call_count == mock_settings.max_retries
            # Verify exponential backoff calls (should be called max_retries - 1 times)
            expected_sleep_calls = [
                call(2**i) for i in range(mock_settings.max_retries - 1)]
            mock_sleep.assert_has_calls(expected_sleep_calls)

    @pytest.mark.asyncio
    async def test_predict_timeout_exception(self, client, mock_settings):
        """Test prediction with timeout exception."""
        service_name = 'test-service'
        data = {'input': 'test'}

        with patch('httpx.AsyncClient') as mock_client, \
                patch('asyncio.sleep', new_callable=AsyncMock):

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            result = await client.predict(service_name, data)

            assert result is None
            assert mock_client.return_value.__aenter__.return_value.post.call_count == mock_settings.max_retries

    @pytest.mark.asyncio
    async def test_predict_connect_error(self, client, mock_settings):
        """Test prediction with connection error."""
        service_name = 'test-service'
        data = {'input': 'test'}

        with patch('httpx.AsyncClient') as mock_client, \
                patch('asyncio.sleep', new_callable=AsyncMock):

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )

            result = await client.predict(service_name, data)

            assert result is None
            assert mock_client.return_value.__aenter__.return_value.post.call_count == mock_settings.max_retries

    @pytest.mark.asyncio
    async def test_predict_generic_exception(self, client, mock_settings):
        """Test prediction with generic exception."""
        service_name = 'test-service'
        data = {'input': 'test'}

        with patch('httpx.AsyncClient') as mock_client, \
                patch('asyncio.sleep', new_callable=AsyncMock):

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Generic error")
            )

            result = await client.predict(service_name, data)

            assert result is None
            assert mock_client.return_value.__aenter__.return_value.post.call_count == mock_settings.max_retries

    @pytest.mark.asyncio
    async def test_predict_success_after_retry(self, client, mock_settings):
        """Test successful prediction after initial failure."""
        service_name = 'test-service'
        data = {'input': 'test'}
        expected_response = {'status': 'normal', 'mae': 0.5}

        with patch('httpx.AsyncClient') as mock_client, \
                patch('asyncio.sleep', new_callable=AsyncMock):

            # First call fails, second succeeds
            mock_responses = [
                Mock(status_code=500),
                Mock(status_code=200, json=Mock(return_value=expected_response))
            ]

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=mock_responses)

            result = await client.predict(service_name, data)

            assert result == expected_response
            assert mock_client.return_value.__aenter__.return_value.post.call_count == 2

    # Tests for predict_welding_data method
    @pytest.mark.asyncio
    async def test_predict_welding_data_success(self, client, mock_settings, sample_data):
        """Test successful welding data prediction."""
        current_result = {'status': 'normal', 'mae': 0.1, 'threshold': 0.5}
        vibration_result = {'status': 'normal', 'mae': 0.2, 'threshold': 0.6}

        with patch.object(client, '_single_predict', new_callable=AsyncMock) as mock_single_predict:
            mock_single_predict.side_effect = [
                current_result, vibration_result]

            result = await client.predict_welding_data(
                sample_data['current_data'],
                sample_data['vibration_data']
            )

            assert result['current'] == current_result
            assert result['vibration'] == vibration_result
            assert result['combined']['status'] == 'normal'
            assert mock_single_predict.call_count == 2

            # Verify correct URLs were called
            expected_url = 'http://localhost:8001/api/predict'
            mock_single_predict.assert_any_call(
                expected_url, sample_data['current_data'], '전류')
            mock_single_predict.assert_any_call(
                expected_url, sample_data['vibration_data'], '진동')

    @pytest.mark.asyncio
    async def test_predict_welding_data_unknown_service(self, client, sample_data):
        """Test welding data prediction with unknown service."""
        with patch('app.services.model_client.settings') as mock_settings:
            mock_settings.model_services = {}

            result = await client.predict_welding_data(
                sample_data['current_data'],
                sample_data['vibration_data']
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_predict_welding_data_partial_failure(self, client, mock_settings, sample_data):
        """Test welding data prediction with partial failure."""
        current_result = {'status': 'normal', 'mae': 0.1, 'threshold': 0.5}
        vibration_result = None

        with patch.object(client, '_single_predict', new_callable=AsyncMock) as mock_single_predict:
            mock_single_predict.side_effect = [
                current_result, vibration_result]

            result = await client.predict_welding_data(
                sample_data['current_data'],
                sample_data['vibration_data']
            )

            assert result['current'] == current_result
            assert result['vibration'] is None
            assert result['combined']['status'] == 'error'

    # Tests for _single_predict method
    @pytest.mark.asyncio
    async def test_single_predict_success(self, client, mock_settings):
        """Test successful single prediction."""
        predict_url = 'http://localhost:8001/api/predict'
        data = {'input': 'test'}
        data_type = '전류'
        expected_response = {'status': 'normal', 'mae': 0.5}

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response)

            result = await client._single_predict(predict_url, data, data_type)

            assert result == expected_response

    @pytest.mark.asyncio
    async def test_single_predict_all_failures(self, client, mock_settings):
        """Test single prediction with all attempts failing."""
        predict_url = 'http://localhost:8001/api/predict'
        data = {'input': 'test'}
        data_type = '전류'

        with patch('httpx.AsyncClient') as mock_client, \
                patch('asyncio.sleep', new_callable=AsyncMock):

            mock_response = Mock()
            mock_response.status_code = 500

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response)

            result = await client._single_predict(predict_url, data, data_type)

            assert result is None
            assert mock_client.return_value.__aenter__.return_value.post.call_count == mock_settings.max_retries

    # Tests for _combine_results method
    def test_combine_results_both_normal(self, client):
        """Test combining results when both are normal."""
        current_result = {'status': 'normal', 'mae': 0.1, 'threshold': 0.5}
        vibration_result = {'status': 'normal', 'mae': 0.2, 'threshold': 0.6}

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'normal'
        assert result['current']['status'] == 'normal'
        assert result['current']['mae'] == 0.1
        assert result['current']['threshold'] == 0.5
        assert result['vibration']['status'] == 'normal'
        assert result['vibration']['mae'] == 0.2
        assert result['vibration']['threshold'] == 0.6
        assert '전류: normal, 진동: normal → 최종: normal' in result['combined_logic']

    def test_combine_results_current_anomaly(self, client):
        """Test combining results when current data shows anomaly."""
        current_result = {'status': 'anomaly', 'mae': 0.8, 'threshold': 0.5}
        vibration_result = {'status': 'normal', 'mae': 0.2, 'threshold': 0.6}

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'anomaly'
        assert result['current']['status'] == 'anomaly'
        assert result['vibration']['status'] == 'normal'
        assert '전류: anomaly, 진동: normal → 최종: anomaly' in result['combined_logic']

    def test_combine_results_vibration_anomaly(self, client):
        """Test combining results when vibration data shows anomaly."""
        current_result = {'status': 'normal', 'mae': 0.1, 'threshold': 0.5}
        vibration_result = {'status': 'anomaly', 'mae': 0.8, 'threshold': 0.6}

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'anomaly'
        assert result['current']['status'] == 'normal'
        assert result['vibration']['status'] == 'anomaly'
        assert '전류: normal, 진동: anomaly → 최종: anomaly' in result['combined_logic']

    def test_combine_results_both_anomaly(self, client):
        """Test combining results when both show anomaly."""
        current_result = {'status': 'anomaly', 'mae': 0.8, 'threshold': 0.5}
        vibration_result = {'status': 'anomaly', 'mae': 0.9, 'threshold': 0.6}

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'anomaly'
        assert result['current']['status'] == 'anomaly'
        assert result['vibration']['status'] == 'anomaly'
        assert '전류: anomaly, 진동: anomaly → 최종: anomaly' in result['combined_logic']

    def test_combine_results_current_none(self, client):
        """Test combining results when current result is None."""
        current_result = None
        vibration_result = {'status': 'normal', 'mae': 0.2, 'threshold': 0.6}

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'error'
        assert result['current_status'] == 'error'
        assert result['vibration_status'] == 'normal'
        assert 'message' in result
        assert '일부 예측 결과를 받을 수 없습니다.' in result['message']

    def test_combine_results_vibration_none(self, client):
        """Test combining results when vibration result is None."""
        current_result = {'status': 'normal', 'mae': 0.1, 'threshold': 0.5}
        vibration_result = None

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'error'
        assert result['current_status'] == 'normal'
        assert result['vibration_status'] == 'error'

    def test_combine_results_both_none(self, client):
        """Test combining results when both results are None."""
        current_result = None
        vibration_result = None

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'error'
        assert result['current_status'] == 'error'
        assert result['vibration_status'] == 'error'

    def test_combine_results_missing_status_keys(self, client):
        """Test combining results when status keys are missing."""
        current_result = {'mae': 0.1, 'threshold': 0.5}  # No status
        vibration_result = {'mae': 0.2, 'threshold': 0.6}  # No status

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'normal'  # Default to normal
        assert result['current']['status'] == 'normal'
        assert result['vibration']['status'] == 'normal'

    def test_combine_results_missing_mae_threshold(self, client):
        """Test combining results when mae/threshold keys are missing."""
        current_result = {'status': 'normal'}  # No mae, threshold
        vibration_result = {'status': 'anomaly'}  # No mae, threshold

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'anomaly'
        assert result['current']['mae'] is None
        assert result['current']['threshold'] is None
        assert result['vibration']['mae'] is None
        assert result['vibration']['threshold'] is None

    # Tests for health_check method
    @pytest.mark.asyncio
    async def test_health_check_success(self, client, mock_settings):
        """Test successful health check."""
        service_name = 'test-service'

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response)

            result = await client.health_check(service_name)

            assert result is True
            mock_client.return_value.__aenter__.return_value.get.assert_called_once_with(
                'http://localhost:8002/health'
            )

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client, mock_settings):
        """Test health check failure."""
        service_name = 'test-service'

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 500

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response)

            result = await client.health_check(service_name)

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_unknown_service(self, client, mock_settings):
        """Test health check for unknown service."""
        result = await client.health_check('unknown-service')
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self, client, mock_settings):
        """Test health check with exception."""
        service_name = 'test-service'

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection error")
            )

            result = await client.health_check(service_name)

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout_setting(self, client, mock_settings):
        """Test health check uses correct timeout setting."""
        service_name = 'test-service'

        with patch('httpx.AsyncClient') as mock_client, \
                patch('httpx.Timeout') as mock_timeout:

            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response)

            await client.health_check(service_name)

            # Verify that a 5.0 second timeout is used for health checks
            mock_timeout.assert_called_with(5.0)

    # Tests for health_check_all method
    @pytest.mark.asyncio
    async def test_health_check_all_success(self, client, mock_settings):
        """Test successful health check for all services."""
        with patch.object(client, 'health_check', new_callable=AsyncMock) as mock_health_check:
            mock_health_check.return_value = True

            result = await client.health_check_all()

            assert len(result) == 2
            assert result['welding-machine'] is True
            assert result['test-service'] is True
            assert mock_health_check.call_count == 2

    @pytest.mark.asyncio
    async def test_health_check_all_mixed_results(self, client, mock_settings):
        """Test health check for all services with mixed results."""
        with patch.object(client, 'health_check', new_callable=AsyncMock) as mock_health_check:
            mock_health_check.side_effect = [True, False]

            result = await client.health_check_all()

            assert len(result) == 2
            assert result['welding-machine'] is True
            assert result['test-service'] is False

    @pytest.mark.asyncio
    async def test_health_check_all_with_exceptions(self, client, mock_settings):
        """Test health check for all services with exceptions."""
        with patch.object(client, 'health_check', new_callable=AsyncMock) as mock_health_check:
            mock_health_check.side_effect = [True, Exception("Error")]

            result = await client.health_check_all()

            assert len(result) == 2
            assert result['welding-machine'] is True
            assert result['test-service'] is False

    @pytest.mark.asyncio
    async def test_health_check_all_empty_services(self, client):
        """Test health check when no services are configured."""
        with patch('app.services.model_client.settings') as mock_settings:
            mock_settings.model_services = {}

            result = await client.health_check_all()

            assert result == {}

    @pytest.mark.asyncio
    async def test_health_check_all_concurrent_execution(self, client, mock_settings):
        """Test that health_check_all executes checks concurrently."""
        with patch.object(client, 'health_check', new_callable=AsyncMock) as mock_health_check, \
                patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:

            mock_health_check.return_value = True
            mock_gather.return_value = [True, True]

            result = await client.health_check_all()

            # Verify asyncio.gather was called for concurrent execution
            mock_gather.assert_called_once()
            assert len(result) == 2

    # Integration-style tests
    @pytest.mark.asyncio
    async def test_full_welding_prediction_workflow(self, client, mock_settings, sample_data):
        """Integration test for complete welding prediction workflow."""
        current_result = {'status': 'normal', 'mae': 0.1, 'threshold': 0.5}
        vibration_result = {'status': 'anomaly', 'mae': 0.8, 'threshold': 0.6}

        with patch('httpx.AsyncClient') as mock_client:
            mock_responses = [
                Mock(status_code=200, json=Mock(return_value=current_result)),
                Mock(status_code=200, json=Mock(return_value=vibration_result))
            ]

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=mock_responses)

            result = await client.predict_welding_data(
                sample_data['current_data'],
                sample_data['vibration_data']
            )

            # Verify full workflow
            assert result['current'] == current_result
            assert result['vibration'] == vibration_result
            # Any anomaly makes final anomaly
            assert result['combined']['status'] == 'anomaly'
            assert '전류: normal, 진동: anomaly → 최종: anomaly' in result['combined']['combined_logic']

    # Edge cases and boundary tests
    @pytest.mark.asyncio
    async def test_predict_with_empty_data(self, client, mock_settings):
        """Test prediction with empty data."""
        service_name = 'test-service'
        data = {}
        expected_response = {'status': 'normal'}

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response)

            result = await client.predict(service_name, data)

            assert result == expected_response

    @pytest.mark.asyncio
    async def test_predict_with_large_data(self, client, mock_settings):
        """Test prediction with large data payload."""
        service_name = 'test-service'
        data = {'values': list(range(1000))}  # Large data
        expected_response = {'status': 'normal'}

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response)

            result = await client.predict(service_name, data)

            assert result == expected_response

    @pytest.mark.asyncio
    async def test_predict_various_http_status_codes(self, client, mock_settings):
        """Test prediction with various HTTP status codes."""
        service_name = 'test-service'
        data = {'input': 'test'}

        status_codes_to_test = [201, 400, 401, 403, 404, 500, 502, 503]

        for status_code in status_codes_to_test:
            with patch('httpx.AsyncClient') as mock_client, \
                    patch('asyncio.sleep', new_callable=AsyncMock):

                mock_response = Mock()
                mock_response.status_code = status_code

                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response)

                result = await client.predict(service_name, data)

                # All non-200 codes should result in None after retries
                assert result is None

    @pytest.mark.asyncio
    async def test_predict_json_decode_error(self, client, mock_settings):
        """Test prediction when response JSON is malformed."""
        service_name = 'test-service'
        data = {'input': 'test'}

        with patch('httpx.AsyncClient') as mock_client, \
                patch('asyncio.sleep', new_callable=AsyncMock):

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError(
                "Invalid JSON", "", 0)

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response)

            result = await client.predict(service_name, data)

            # JSON decode error should be caught as generic exception
            assert result is None

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, client, mock_settings):
        """Test that exponential backoff follows correct timing pattern."""
        service_name = 'test-service'
        data = {'input': 'test'}

        with patch('httpx.AsyncClient') as mock_client, \
                patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            await client.predict(service_name, data)

            # Verify exponential backoff: 2^0=1, 2^1=2 for 3 retries
            expected_calls = [call(1), call(2)]
            mock_sleep.assert_has_calls(expected_calls)

    # Test global instance
    def test_global_model_client_instance(self):
        """Test that global model_client instance is properly created."""
        assert isinstance(model_client, ModelClient)
        assert hasattr(model_client, 'timeout')
        assert hasattr(model_client, 'max_retries')

    @pytest.mark.asyncio
    async def test_timeout_configuration(self, client, mock_settings):
        """Test that timeout configuration is properly applied."""
        service_name = 'test-service'
        data = {'input': 'test'}

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'success'}

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response)

            await client.predict(service_name, data)

            # Verify timeout was passed to AsyncClient
            mock_client.assert_called_with(timeout=client.timeout)

    def test_combine_results_with_extra_fields(self, client):
        """Test combining results with extra fields."""
        current_result = {
            'status': 'normal',
            'mae': 0.1,
            'threshold': 0.5,
            'extra_field': 'value'
        }
        vibration_result = {
            'status': 'normal',
            'mae': 0.2,
            'threshold': 0.6,
            'another_field': 'another_value'
        }

        result = client._combine_results(current_result, vibration_result)

        assert result['status'] == 'normal'
        assert result['current']['mae'] == 0.1
        assert result['vibration']['mae'] == 0.2
        # Extra fields should not affect the combination logic

    @pytest.mark.asyncio
    async def test_service_url_formatting(self, client):
        """Test that service URLs are properly formatted."""
        with patch('app.services.model_client.settings') as mock_settings:
            mock_settings.model_services = {
                'service1': 'http://localhost:8001',      # No trailing slash
                'service2': 'http://localhost:8002/',     # With trailing slash
                'service3': 'https://example.com:443/api'  # Complex URL
            }

            # Test service1
            with patch('httpx.AsyncClient') as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=Mock(
                        status_code=200, json=Mock(return_value={}))
                )

                await client.predict('service1', {})

                # Should append /api/predict correctly
                mock_client.return_value.__aenter__.return_value.post.assert_called_with(
                    'http://localhost:8001/api/predict', json={}
                )

    @pytest.mark.asyncio
    async def test_httpx_client_context_manager_usage(self, client, mock_settings):
        """Test that httpx AsyncClient is properly used as context manager."""
        service_name = 'test-service'
        data = {'input': 'test'}

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'success'}

            mock_client.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response)

            await client.predict(service_name, data)

            # Verify context manager usage
            mock_client.__aenter__.assert_called_once()
            mock_client.__aexit__.assert_called_once()
