import asyncio
from typing import Any, Dict

import pytest

# Note on test framework:
# These tests use pytest with pytest.mark.asyncio to validate async behavior without adding new dependencies.

# Import the service under test and its dependencies
from app.services.prediction_api_service import PredictAPIService  # assuming service path; adjust if different
from app.models.data_models import PredictionRequest, PredictionResult
from app.config import settings as settings_module


def make_prediction_request(
    n_ai0: int = 10, n_ai1: int = 12, n_ai2: int = 8
) -> PredictionRequest:
    # Build a minimal valid PredictionRequest object.
    # The exact model fields are unknown from the diff; we infer from debug log keys used:
    #  - AI0_Vibration: sequence-like
    #  - AI1_Vibration: sequence-like
    #  - AI2_Current:   sequence-like
    # If additional required fields exist, please update accordingly to match the model.
    data = {
        "AI0_Vibration": [0.1] * n_ai0,
        "AI1_Vibration": [0.2] * n_ai1,
        "AI2_Current": [0.3] * n_ai2,
    }
    # PredictionRequest is likely a Pydantic model; construct it via its constructor.
    return PredictionRequest(**data)


class DummyResponse:
    def __init__(self, status: int, json_payload: Dict[str, Any] | None = None, text_body: str = ""):
        self.status = status
        self._json_payload = json_payload
        self._text_body = text_body

    async def json(self):
        # Simulate aiohttp response.json()
        if self._json_payload is None:
            raise ValueError("No JSON payload")
        return self._json_payload

    async def text(self):
        return self._text_body

    # Context manager protocol for "async with"
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummySession:
    def __init__(self, timeout=None, post_side_effect=None, get_side_effect=None):
        self._timeout = timeout
        self._post_side_effect = post_side_effect
        self._get_side_effect = get_side_effect
        self.post_calls = []
        self.get_calls = []

    # Context manager protocol for "async with aiohttp.ClientSession(...) as session"
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        # aiohttp returns an async context manager; we emulate any side effect or return a DummyResponse
        self.post_calls.append({"url": url, "json": json, "headers": headers})
        if callable(self._post_side_effect):
            result = self._post_side_effect(url, json, headers)
            return result
        # Default: return 200 ok with reasonable payload
        return DummyResponse(
            200,
            {
                "prediction": "OK",
                "is_fault": False,
            },
        )

    def get(self, url):
        self.get_calls.append({"url": url})
        if callable(self._get_side_effect):
            return self._get_side_effect(url)
        return DummyResponse(200, None, "healthy")


class DummyTimeoutError(Exception):
    pass


@pytest.fixture
def patch_sleep(monkeypatch):
    # Speed up retries by replacing asyncio.sleep with a no-op that still yields control
    async def fast_sleep(_):
        await asyncio.sleep(0)  # one loop to maintain async semantics
    # To avoid recursion, patch to a coroutine that just returns None without calling itself
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    yield
    # No explicit unpatch needed as monkeypatch reverts automatically


@pytest.fixture
def patch_aiohttp_session(monkeypatch):
    # Patch aiohttp.ClientSession to return our DummySession
    import aiohttp

    def _apply(session_factory):
        class FactorySession:
            def __init__(self, timeout=None):
                self._timeout = timeout
                self._session = session_factory(timeout)

            async def __aenter__(self):
                return self._session

            async def __aexit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr(aiohttp, "ClientSession", FactorySession)
        return FactorySession

    return _apply


@pytest.fixture
def patch_settings(monkeypatch):
    # Patch URLs used by the service
    monkeypatch.setattr(settings_module.settings, "PREDICTION_API_FULL_URL", "http://example.test/predict", raising=False)
    monkeypatch.setattr(settings_module.settings, "PRESS_FAULT_MODEL_BASE_URL", "http://example.test", raising=False)


@pytest.mark.asyncio
async def test_call_predict_api_success_happy_path(patch_settings, patch_aiohttp_session):
    # Arrange: 200 response with valid JSON that matches PredictionResult
    def session_factory(_timeout):
        def post_side_effect(url, payload, headers):
            # Validate that JSON matches request.model_dump() keys
            assert "AI0_Vibration" in payload and "AI1_Vibration" in payload and "AI2_Current" in payload
            # Return a context manager which yields a DummyResponse
            return DummyResponse(
                200,
                {
                    # Minimal plausible fields; adjust if PredictionResult requires more
                    "prediction": "NORMAL",
                    "is_fault": False,
                },
            )
        return DummySession(timeout=_timeout, post_side_effect=post_side_effect)

    patch_aiohttp_session(session_factory)

    svc = PredictAPIService()
    req = make_prediction_request()

    # Act
    result = await svc.call_precict_api(req)

    # Assert
    assert isinstance(result, PredictionResult)
    assert result.prediction in ("NORMAL", "OK")
    assert result.is_fault is False


@pytest.mark.asyncio
async def test_call_predict_api_invalid_response_returns_none(patch_settings, patch_aiohttp_session):
    # Arrange: 200 but invalid JSON payload that breaks PredictionResult construction
    def session_factory(_timeout):
        def post_side_effect(url, payload, headers):
            # Missing required fields to force a validation error
            return DummyResponse(200, {"unexpected": "data"})
        return DummySession(timeout=_timeout, post_side_effect=post_side_effect)

    patch_aiohttp_session(session_factory)

    svc = PredictAPIService()
    req = make_prediction_request()

    # Act
    result = await svc.call_precict_api(req)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_call_predict_api_non_200_retries_then_returns_none(
    patch_settings, patch_aiohttp_session, patch_sleep, monkeypatch
):
    attempts = []

    def session_factory(_timeout):
        def post_side_effect(url, payload, headers):
            attempts.append(1)
            # Always return 500 with some text body
            return DummyResponse(500, None, "Internal Server Error")
        return DummySession(timeout=_timeout, post_side_effect=post_side_effect)

    patch_aiohttp_session(session_factory)

    svc = PredictAPIService()
    # Reduce max_retries to 2 for faster test while still exercising retry logic
    monkeypatch.setattr(svc, "max_retries", 2)

    req = make_prediction_request()

    result = await svc.call_precict_api(req)

    assert result is None
    # Should have attempted exactly max_retries times
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_call_predict_api_timeout_error_retries_then_none(
    patch_settings, patch_aiohttp_session, patch_sleep, monkeypatch
):
    attempts = []

    class RaisingPost:
        def __init__(self):
            self._count = 0

        # This is returned by session.post and acts as async context manager
        async def __aenter__(self):
            raise asyncio.TimeoutError("timed out")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def session_factory(_timeout):
        def post_side_effect(url, payload, headers):
            attempts.append(1)
            return RaisingPost()
        return DummySession(timeout=_timeout, post_side_effect=post_side_effect)

    patch_aiohttp_session(session_factory)

    svc = PredictAPIService()
    monkeypatch.setattr(svc, "max_retries", 3)

    req = make_prediction_request()

    result = await svc.call_precict_api(req)

    assert result is None
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_call_predict_api_client_connection_error_retries_then_none(
    patch_settings, patch_aiohttp_session, patch_sleep, monkeypatch
):
    attempts = []

    class RaisingPost:
        async def __aenter__(self):
            import aiohttp
            raise aiohttp.ClientConnectionError("connection failed")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def session_factory(_timeout):
        def post_side_effect(url, payload, headers):
            attempts.append(1)
            return RaisingPost()
        return DummySession(timeout=_timeout, post_side_effect=post_side_effect)

    patch_aiohttp_session(session_factory)

    svc = PredictAPIService()
    monkeypatch.setattr(svc, "max_retries", 2)

    req = make_prediction_request()

    result = await svc.call_precict_api(req)

    assert result is None
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_call_predict_api_unexpected_exception_retries_then_none(
    patch_settings, patch_aiohttp_session, patch_sleep, monkeypatch
):
    attempts = []

    class RaisingPost:
        async def __aenter__(self):
            raise RuntimeError("unexpected")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def session_factory(_timeout):
        def post_side_effect(url, payload, headers):
            attempts.append(1)
            return RaisingPost()
        return DummySession(timeout=_timeout, post_side_effect=post_side_effect)

    patch_aiohttp_session(session_factory)

    svc = PredictAPIService()
    monkeypatch.setattr(svc, "max_retries", 2)

    req = make_prediction_request()

    result = await svc.call_precict_api(req)

    assert result is None
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_health_check_success_returns_true(patch_settings, patch_aiohttp_session):
    # Arrange: GET /health returns 200
    def session_factory(_timeout):
        def get_side_effect(url):
            assert url.endswith("/health")
            return DummyResponse(200, None, "healthy")
        return DummySession(timeout=_timeout, get_side_effect=get_side_effect)

    patch_aiohttp_session(session_factory)

    svc = PredictAPIService()

    ok = await svc.health_check()
    assert ok is True


@pytest.mark.asyncio
async def test_health_check_non_200_returns_false(patch_settings, patch_aiohttp_session):
    # Arrange: GET /health returns 503
    def session_factory(_timeout):
        def get_side_effect(url):
            return DummyResponse(503, None, "unhealthy")
        return DummySession(timeout=_timeout, get_side_effect=get_side_effect)

    patch_aiohttp_session(session_factory)

    svc = PredictAPIService()

    ok = await svc.health_check()
    assert ok is False


@pytest.mark.asyncio
async def test_health_check_requests_exception_caught_returns_false(patch_settings, monkeypatch):
    # The code catches requests.exceptions.RequestException despite using aiohttp.
    # We simulate that specific exception to ensure the except block is executed.
    import requests
    import aiohttp

    class RaisingGetSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            class Ctx:
                async def __aenter__(self_inner):
                    raise requests.exceptions.RequestException("simulated requests exception")

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False
            return Ctx()

    class SessionFactory:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return RaisingGetSession()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(aiohttp, "ClientSession", SessionFactory)

    svc = PredictAPIService()
    ok = await svc.health_check()
    assert ok is False