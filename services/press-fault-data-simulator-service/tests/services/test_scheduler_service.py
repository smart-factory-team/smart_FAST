import asyncio
from contextlib import suppress
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Testing library and framework: pytest with pytest-asyncio for async test support
# We rely on pytest's fixtures and monkeypatch for mocking and isolation.

# We import the class under test. Adjust import path if the module is placed differently.
# The service code references:
#   from app.services.azure_storage_service import AzureStorageService
#   from app.services.prediction_api_service import PredictAPIService
#   from app.models.data_models import PredictionRequest
#   from app.config.settings import settings
#   from app.utils.logger import system_log, prediction_log
#
# By default, we import SchedulerService from app.services.scheduler_service (common path),
# but if project structure differs, update this path accordingly.
try:
    from app.services.scheduler_service import SchedulerService
except Exception as e:  # pragma: no cover - guidance for maintainers if path differs
    raise ImportError(
        "Could not import SchedulerService from app.services.scheduler_service. "
        "If your module path differs, update the import in tests/services/test_scheduler_service.py"
    ) from e


@pytest.fixture
def event_loop():
    # Use a fresh event loop to isolate tests and ensure proper cleanup of async tasks.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    # Cancel any pending tasks and shutdown async generators in a 3.11-compatible way
    async def _cancel_pending():
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    with suppress(Exception):
        loop.run_until_complete(_cancel_pending())
        loop.run_until_complete(loop.shutdown_asyncgens())
    asyncio.set_event_loop(None)
    loop.close()


@pytest.fixture
def make_service(monkeypatch):
    """
    Factory fixture to build a SchedulerService with mocked dependencies and controllable behavior.
    We patch AzureStorageService, PredictAPIService, settings, and loggers.
    """
    def _make(
        *,
        api_health=True,
        storage_connect=True,
        single_sim_ok=True,
        next_minute_data=None,
        predict_result=None,
        end_of_file=False,
        fault=False,
        fault_probability=0.42,
        api_call_raises=None,
        storage_get_raises=None,
        interval_minutes=0  # we'll patch sleep anyway; 0 keeps runtime short
    ):
        # Build mocks
        mock_storage = MagicMock(name="AzureStorageServiceMock")
        mock_api = MagicMock(name="PredictAPIServiceMock")

        mock_storage.connect = AsyncMock(return_value=storage_connect)
        if storage_get_raises is not None:
            mock_storage.get_next_minute_data = AsyncMock(side_effect=storage_get_raises)
        else:
            if next_minute_data is None:
                # default minute_data placeholder
                minute_data = {"row": ["1", "2", "3"]}
                file_name = "demo.csv"
                is_end = end_of_file
                next_minute_data = (minute_data, file_name, is_end)
            mock_storage.get_next_minute_data = AsyncMock(return_value=next_minute_data)
        mock_storage.close = AsyncMock(return_value=None)
        mock_storage.get_current_status = MagicMock(return_value={"connected": storage_connect})

        mock_api.health_check = AsyncMock(return_value=api_health)
        if api_call_raises is not None:
            mock_api.call_precict_api = AsyncMock(side_effect=api_call_raises)
        else:
            if predict_result is None:
                predict_result = SimpleNamespace(
                    prediction="OK" if not fault else "FAULT",
                    is_fault=bool(fault),
                    fault_probability=fault_probability if fault_probability is not None else None,
                )
            mock_api.call_precict_api = AsyncMock(return_value=predict_result)

        # Patch constructors to return our mocks
        monkeypatch.setattr("app.services.scheduler_service.AzureStorageService", lambda: mock_storage)
        monkeypatch.setattr("app.services.scheduler_service.PredictAPIService", lambda: mock_api)

        # Patch PredictionRequest to avoid real parsing; just echo input
        class FakePredictionRequest:
            def __init__(self, data):
                self.parsed = True
                self.data = data
            @classmethod
            def from_csv_data(cls, data):
                return cls(data)
            # Minimal Pydantic compatibility
            def model_dump(self):
                return {"parsed": self.parsed, "data": self.data}
            def dict(self):
                return self.model_dump()
        monkeypatch.setattr("app.services.scheduler_service.PredictionRequest", FakePredictionRequest)
        
        # Patch settings and logs
        fake_settings = SimpleNamespace(
            SIMULATOR_INTERVAL_MINUTES=interval_minutes,
            PREDICTION_API_FULL_URL="http://localhost/predict",
        )
        monkeypatch.setattr("app.services.scheduler_service.settings", fake_settings, raising=True)

        # Logs: make them MagicMocks so we can assert messages indirectly if needed
        fake_system_log = MagicMock(name="system_log")
        fake_prediction_log = MagicMock(name="prediction_log")
        monkeypatch.setattr("app.services.scheduler_service.system_log", fake_system_log, raising=True)
        monkeypatch.setattr("app.services.scheduler_service.prediction_log", fake_prediction_log, raising=True)

        service = SchedulerService()
        return service, mock_storage, mock_api, fake_system_log, fake_prediction_log, fake_settings

    return _make


@pytest.mark.asyncio
async def test_start_simulation_success_starts_loop_and_initializes_stats(make_service):
    service, storage, api, sys_log, pred_log, cfg = make_service(
        api_health=True, storage_connect=True, interval_minutes=0
    )

    # Patch asyncio.create_task path used by service: it calls loop.create_task
    # We ensure it creates a task but don't run the loop indefinitely.
    # Also, patch asyncio.sleep inside the loop to avoid real sleeping.
    with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
        started = await service.start_simulation()

    assert started is True
    assert service.is_running is True
    assert service.start_time is not None
    assert isinstance(service.start_time, datetime)
    assert service.total_predictions == 0
    assert service.fault_detections == 0
    sys_log.info.assert_any_call("🚀 시뮬레이션 시작 중...")
    # Confirm that the background task was scheduled
    assert hasattr(service, "task")
    assert asyncio.isfuture(service.task) or asyncio.iscoroutine(service.task)


@pytest.mark.asyncio
async def test_start_simulation_returns_false_if_already_running(make_service):
    service, *_ = make_service()
    service.is_running = True

    started = await service.start_simulation()
    assert started is False


@pytest.mark.asyncio
async def test_start_simulation_fails_on_api_health_check(make_service):
    service, storage, api, sys_log, _, _ = make_service(api_health=False)

    started = await service.start_simulation()
    assert started is False
    sys_log.error.assert_any_call("API 서버 연결 실패. 시뮬레이션을 시작할 수 없습니다.")
    assert service.is_running is False


@pytest.mark.asyncio
async def test_start_simulation_fails_on_storage_connect(make_service):
    service, storage, api, sys_log, _, _ = make_service(api_health=True, storage_connect=False)

    started = await service.start_simulation()
    assert started is False
    sys_log.error.assert_any_call("Azure Storage 연결 실패.")
    assert service.is_running is False


@pytest.mark.asyncio
async def test_stop_simulation_graceful_when_not_running(make_service):
    service, storage, api, sys_log, _, _ = make_service()
    service.is_running = False

    stopped = await service.stop_simulation()
    assert stopped is False
    sys_log.warning.assert_any_call("시뮬레이션이 실행 중이지 않습니다.")


@pytest.mark.asyncio
async def test_stop_simulation_cancels_task_and_closes_storage(make_service):
    service, storage, api, sys_log, _, _ = make_service()

    with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
        await service.start_simulation()
    # Ensure task exists and is pending
    assert hasattr(service, "task")
    assert not service.task.done()

    # Stop should cancel the task and close storage
    with patch("asyncio.wait_for", new=AsyncMock(return_value=None)):
        stopped = await service.stop_simulation()

    assert stopped is True
    assert service.is_running is False
    storage.close.assert_awaited()
    sys_log.info.assert_any_call("✅ 시뮬레이션 종료 완료")


@pytest.mark.asyncio
async def test_run_single_simulation_happy_path_updates_stats_and_logs(make_service):
    # Simulate a normal (non-fault) prediction
    service, storage, api, sys_log, pred_log, _ = make_service(fault=False, fault_probability=0.1234)

    ok = await service._run_single_simulation()
    assert ok is True
    assert service.total_predictions == 1
    assert service.fault_detections == 0
    # Ensure logging via prediction_log.info for normal
    assert pred_log.info.call_count >= 1
    # No warning for normal case
    assert pred_log.warning.call_count == 0


@pytest.mark.asyncio
async def test_run_single_simulation_fault_increments_fault_count_and_warns(make_service):
    # Simulate a fault prediction
    service, storage, api, sys_log, pred_log, _ = make_service(fault=True, fault_probability=0.9)

    ok = await service._run_single_simulation()
    assert ok is True
    assert service.total_predictions == 1
    assert service.fault_detections == 1
    # Ensure warning for fault
    assert pred_log.warning.call_count >= 1
    assert pred_log.info.call_count == 0


@pytest.mark.asyncio
async def test_run_single_simulation_handles_none_result_from_storage(make_service):
    service, storage, api, sys_log, _, _ = make_service()
    # Override get_next_minute_data to return None
    storage.get_next_minute_data.return_value = None

    ok = await service._run_single_simulation()
    assert ok is False
    sys_log.warning.assert_any_call("데이터를 가져올 수 없습니다.")
    # Counters should not increment
    assert service.total_predictions == 0
    assert service.fault_detections == 0


@pytest.mark.asyncio
async def test_run_single_simulation_handles_api_returning_none(make_service):
    service, storage, api, sys_log, _, _ = make_service()
    api.call_precict_api.return_value = None

    ok = await service._run_single_simulation()
    assert ok is False
    sys_log.error.assert_any_call("API 호출 실패")
    assert service.total_predictions == 0
    assert service.fault_detections == 0


@pytest.mark.asyncio
async def test_run_single_simulation_catches_exceptions_and_returns_false(make_service):
    service, storage, api, sys_log, _, _ = make_service(storage_get_raises=RuntimeError("boom"))

    ok = await service._run_single_simulation()
    assert ok is False
    # Should log an error string containing the exception message
    assert any(
        call.args and "시뮬레이션 실행 오류" in call.args[0] and "boom" in call.args[0]
        for call in sys_log.error.mock_calls
    )


def test_handle_prediction_result_formats_probability_and_routes_logs(make_service):
    service, storage, api, sys_log, pred_log, _ = make_service()

    # Fault case with probability None should print "N/A"
    result_fault_none = SimpleNamespace(prediction="FAULT", is_fault=True, fault_probability=None)
    service._handle_prediction_result(result_fault_none, data_source="file.csv")
    # Expect warning called, not info
    assert pred_log.warning.call_count >= 1
    # The message should contain the placeholder "N/A" for probability
    joined_msgs = " | ".join(str(c.args[0]) for c in pred_log.warning.mock_calls if c.args)
    assert "Probability: N/A" in joined_msgs

    pred_log.warning.reset_mock()
    pred_log.info.reset_mock()

    # Normal case with numeric probability
    result_normal = SimpleNamespace(prediction="OK", is_fault=False, fault_probability=0.5)
    service._handle_prediction_result(result_normal, data_source="file.csv")
    assert pred_log.info.call_count >= 1
    assert pred_log.warning.call_count == 0
    joined_info = " | ".join(str(c.args[0]) for c in pred_log.info.mock_calls if c.args)
    assert "✅ NORMAL" in joined_info
    assert "Probability: 0.5000" in joined_info  # 4-decimal formatting


@pytest.mark.asyncio
async def test_get_simulation_status_when_stopped(make_service):
    service, storage, api, sys_log, pred_log, cfg = make_service()
    service.is_running = False
    status = service.get_simulation_status()
    assert status["status"] == "stopped"
    assert status["is_running"] is False
    assert "storage_status" not in status or status["storage_status"] is None


@pytest.mark.asyncio
async def test_get_simulation_status_when_running_includes_metrics(make_service):
    service, storage, api, sys_log, pred_log, cfg = make_service()
    # Simulate started service
    service.is_running = True
    service.start_time = datetime.now() - timedelta(seconds=5)
    service.total_predictions = 3
    service.fault_detections = 1

    status = service.get_simulation_status()
    assert status["status"] == "running"
    assert status["is_running"] is True
    assert status["start_time"] is not None
    assert status["runtime"] is not None
    assert status["total_predictions"] == 3
    assert status["fault_detections"] == 1
    assert status["interval_minutes"] == cfg.SIMULATOR_INTERVAL_MINUTES
    assert status["api_url"] == cfg.PREDICTION_API_FULL_URL
    assert status["storage_status"] == {"connected": True}


@pytest.mark.asyncio
async def test_run_simulation_loop_cancels_on_is_running_false(make_service):
    # Configure a single successful simulation, then flip is_running to False after first sleep iteration
    service, storage, api, sys_log, pred_log, cfg = make_service(interval_minutes=0)

    # Patch sleep to allow loop to proceed without delay
    async def fake_sleep(_):
        # After first simulation call, is_running will be set to False to simulate stop
        service.is_running = False

    with patch("asyncio.sleep", new=AsyncMock(side_effect=fake_sleep)):
        await service.start_simulation()
        # Wait for the task to notice is_running = False and exit loop
        await asyncio.sleep(0)
        # Now stop to ensure cleanup
        with patch("asyncio.wait_for", new=AsyncMock(return_value=None)):
            await service.stop_simulation()

    # Ensure loop start and end logs were called
    assert any("시뮬레이션 루프 시작" in str(c.args[0]) for c in sys_log.info.mock_calls)
    assert any("시뮬레이션 루프 종료" in str(c.args[0]) for c in sys_log.info.mock_calls)


@pytest.mark.asyncio
async def test_run_simulation_loop_handles_exceptions_with_backoff(make_service):
    # Make _run_single_simulation raise an exception to trigger error path and sleep(10)
    service, storage, api, sys_log, pred_log, cfg = make_service(interval_minutes=0)

    async def bad_single():
        raise RuntimeError("loop-failure")

    # Force is_running True for one iteration, then False to exit after handling error
    service.is_running = True
    with patch.object(service, "_run_single_simulation", new=AsyncMock(side_effect=bad_single)), \
         patch("asyncio.sleep", new=AsyncMock(return_value=None)) as sleep_mock:
        # Run loop coroutine directly but only a slice to hit the exception branch once
        coro = service._run_simulation_loop()
        task = asyncio.create_task(coro)
        # Allow it to run and handle the exception once
        await asyncio.sleep(0)
        service.is_running = False
        # Allow cancellation/exit
        await asyncio.sleep(0)
        # Cleanup
        with suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.1)

    # Confirm error logged and that sleep was awaited (backoff path)
    assert any("시뮬레이션 루프 오류" in str(c.args[0]) and "loop-failure" in str(c.args[0]) for c in sys_log.error.mock_calls)
    assert sleep_mock.await_count >= 1