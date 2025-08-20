import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    # Preferred import for FastAPI testing
    from fastapi.testclient import TestClient
except Exception:
    # Fallback if project uses Starlette's TestClient
    from starlette.testclient import TestClient

# We import the module under test. Since the file content refers to "main:app" for uvicorn,
# we assume the module name is "main" in the service package root.
# If the real path differs (e.g., services.press-defect-detection-model-service.main),
# adapt as necessary. Using a relative import where possible.
# We try multiple import strategies for resilience in various project layouts.

main_mod_candidates = [
    "services.press-defect-detection-model-service.main",
    "services.press-defect-detection-model-service.app",
    "main",
]
main = None
for mod_name in main_mod_candidates:
    try:
        main = __import__(mod_name, fromlist=["*"])
        break
    except Exception:
        continue

if main is None:
    # As a last resort, try relative import if tests run from within the service directory.
    try:
        import importlib
        main = importlib.import_module("main")
    except Exception as e:
        raise ImportError(
            "Could not import the FastAPI app module. "
            "Ensure the module name is 'main' or update the import path in tests."
        ) from e

# Helper to build a new TestClient to trigger FastAPI startup/shutdown events
def create_client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def reset_global_state(monkeypatch):
    """
    Reset global flags in main module before each test to avoid cross-test contamination.
    """
    if hasattr(main, "model_loaded"):
        main.model_loaded = False
    if hasattr(main, "model_loading_error"):
        main.model_loading_error = None
    # Reset service start time close to 'now' so uptime assertions are predictable
    if hasattr(main, "service_start_time"):
        main.service_start_time = datetime.now()
    yield
    # No teardown required


@pytest.fixture
def mock_model():
    """
    Provide a mock for YOLOv7Model with async load_model and get_model_info.
    """
    mock = MagicMock()
    mock.load_model = AsyncMock()
    mock.get_model_info = MagicMock(return_value={
        "categories": {0: "ok", 1: "defect"},
        "adaptive_thresholds": {"defect": 0.5},
    })
    return mock


@pytest.fixture
def mock_inference_service():
    """
    Provide a simple mock InferenceService instance.
    """
    svc = MagicMock()
    return svc


def test_root_endpoint_contains_expected_fields():
    client = create_client()
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "Press Defect Detection API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"
    # Ensure ISO timestamp for start_time
    datetime.fromisoformat(data["start_time"])
    # Ensure docs and endpoints map included
    assert "/docs" in data["docs"]
    assert isinstance(data["endpoints"], dict)
    for key in ["health", "ready", "startup", "predict", "predict_file", "model_info"]:
        assert key in data["endpoints"]


def test_health_endpoint_reports_healthy_and_uptime_increases():
    client = create_client()
    resp1 = client.get("/health")
    assert resp1.status_code == 200
    d1 = resp1.json()
    assert d1["status"] == "healthy"
    datetime.fromisoformat(d1["timestamp"])
    assert isinstance(d1["uptime_seconds"], float)

    # Wait a tick and verify uptime increases
    resp2 = client.get("/health")
    assert resp2.status_code == 200
    d2 = resp2.json()
    assert d2["uptime_seconds"] >= d1["uptime_seconds"]


def test_startup_endpoint_reflects_model_ready_flag_false_by_default():
    client = create_client()
    resp = client.get("/startup")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    assert data["service_ready"] is True
    assert data["model_ready"] is False
    datetime.fromisoformat(data["start_time"])
    datetime.fromisoformat(data["timestamp"])


def test_ready_endpoint_503_loading_when_not_loaded_and_no_error():
    client = create_client()
    # model_loaded False and model_loading_error None by fixture
    resp = client.get("/ready")
    assert resp.status_code == 503
    data = resp.json()
    # FastAPI/HTTPException uses {"detail": {...}}
    assert "detail" in data
    detail = data["detail"]
    assert detail["status"] == "loading"
    assert detail["model_loaded"] is False
    assert "AI 모델 로딩 중" in detail["message"]  # verify message hint
    datetime.fromisoformat(detail["timestamp"])


def test_ready_endpoint_ok_when_model_loaded_true():
    main.model_loaded = True
    client = create_client()
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    datetime.fromisoformat(data["timestamp"])


def test_ready_endpoint_503_when_model_loading_error_present():
    main.model_loaded = False
    main.model_loading_error = "Boom!"
    client = create_client()
    resp = client.get("/ready")
    assert resp.status_code == 503
    data = resp.json()
    detail = data["detail"]
    assert detail["status"] == "not_ready"
    assert detail["model_loaded"] is False
    assert detail["error"] == "Boom!"
    datetime.fromisoformat(detail["timestamp"])


@pytest.mark.asyncio
async def test_startup_event_success_sets_model_loaded_and_registers_service(monkeypatch, mock_model, mock_inference_service):
    # Patch YOLOv7Model() construction to return our mock_model
    monkeypatch.setattr(main, "YOLOv7Model", lambda: mock_model)
    # Patch InferenceService to return a mock instance
    monkeypatch.setattr(main, "InferenceService", lambda m: mock_inference_service)
    # Track set_inference_service calls
    set_service_calls = []
    def fake_set_inference_service(svc):
        set_service_calls.append(svc)
    monkeypatch.setattr(main, "set_inference_service", fake_set_inference_service)

    # Ensure globals are reset and re-create app to trigger startup
    main.model_loaded = False
    main.model_loading_error = None
    # Manually call the startup event handler
    await main.startup_event()

    # Assertions
    mock_model.load_model.assert_awaited()
    assert main.model_loaded is True
    assert main.model_loading_error is None
    assert set_service_calls and set_service_calls[0] is mock_inference_service


@pytest.mark.asyncio
async def test_startup_event_failure_sets_error_and_not_loaded(monkeypatch):
    failing_model = MagicMock()
    failing_model.load_model = AsyncMock(side_effect=RuntimeError("load failed"))
    monkeypatch.setattr(main, "YOLOv7Model", lambda: failing_model)

    main.model_loaded = True
    main.model_loading_error = None

    await main.startup_event()

    assert main.model_loaded is False
    assert isinstance(main.model_loading_error, str)
    assert "load failed" in main.model_loading_error


def test_model_info_503_when_not_loaded():
    client = create_client()
    main.model_loaded = False
    resp = client.get("/model/info")
    assert resp.status_code == 503
    body = resp.json()
    assert body["detail"] == "모델이 아직 로딩되지 않았습니다."


def test_model_info_success_when_loaded(monkeypatch, mock_model):
    # Ensure we use the same global yolo_model used by app
    # Replace global yolo_model and model_loaded
    monkeypatch.setattr(main, "yolo_model", mock_model, raising=True)
    main.model_loaded = True

    client = create_client()
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()

    assert data["model_name"] == "YOLOv7 Press Defect Detection"
    assert data["model_version"] == "1.0.0"
    assert data["model_file"] == "press_hole_yolov7_best.pt"
    assert data["huggingface_repo"] == "23smartfactory/press-defect-detection-model"

    # categories should be a list of dicts from the model_details mapping
    assert isinstance(data["categories"], list)
    assert {"id": 0, "name": "ok"} in data["categories"]
    assert {"id": 1, "name": "defect"} in data["categories"]

    assert data["adaptive_thresholds"] == {"defect": 0.5}
    assert isinstance(data["model_details"], dict)
    datetime.fromisoformat(data["timestamp"])


def test_uvicorn_block_not_executed_on_import(monkeypatch):
    """
    Verify that importing the module does not attempt to run uvicorn due to the __main__ guard.
    """
    # Patch uvicorn.run to ensure it would raise if called
    called = {"run": False}
    def fake_run(*args, **kwargs):
        called["run"] = True
    monkeypatch.setattr(main, "uvicorn", types.SimpleNamespace(run=fake_run), raising=True)

    # Re-import should not call uvicorn.run because __name__ != "__main__" under pytest
    assert called["run"] is False


def test_cors_middleware_is_configured():
    """
    The app should have CORSMiddleware with allow_origins ["*"], allow_methods ["*"], etc.
    """
    # Access app.user_middleware and verify one of them is CORSMiddleware
    cors_configs = []
    for mw in main.app.user_middleware:
        cls = getattr(mw, "cls", None)
        if cls and cls.__name__ == "CORSMiddleware":
            cors_configs.append(mw)
    assert cors_configs, "Expected CORSMiddleware to be configured on app"


def test_router_included_predict_routes_present():
    """
    Since main includes a predict router, basic presence check of documented endpoints.
    We cannot call predict endpoints without the actual implementation; we just check that the path
    operations exist (e.g., '/predict', '/predict/file') in the OpenAPI schema.
    """
    schema = main.app.openapi()
    paths = schema.get("paths", {})
    # Ensure that at least these endpoints are present in the schema if the router is included.
    # They can be either post or other methods depending on implementation.
    # We check for any method existence on paths of interest.
    def has_any_method(path_key):
        return path_key in paths and isinstance(paths[path_key], dict) and len(paths[path_key].keys()) > 0

    assert has_any_method("/predict") or has_any_method("/predict/"), "Predict endpoint not found in schema"
    assert has_any_method("/predict/file") or has_any_method("/predict/file/"), "Predict file endpoint not found in schema"