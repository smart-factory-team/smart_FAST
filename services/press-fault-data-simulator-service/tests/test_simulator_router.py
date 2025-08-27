# -*- coding: utf-8 -*-
"""
Unit tests for the simulator router.

Testing library: pytest
Web framework under test: FastAPI (TestClient)
"""

import importlib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _resolve_simulator_router_module():
    """
    Attempts to import the simulator router module from common locations.
    This helps the tests remain robust even if the router path differs slightly.
    """
    candidates = [
        "app.routers.simulator_router",
        "app.routes.simulator_router",
        "app.api.routers.simulator_router",
        "app.api.routes.simulator_router",
        "app.simulator_router",
        "simulator_router",
    ]
    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            last_err = e
            continue
        except Exception:
            # Surface real import-time errors from the target module instead of hiding them
            raise
    tried = ", ".join(candidates)
    raise ImportError(f"Could not import simulator_router from any of: {tried}. Last error: {last_err}")

@pytest.fixture(scope="module")
def module_router_module():
    return _resolve_simulator_router_module()


@pytest.fixture()
def router_test_client(module_router_module):
    app = FastAPI()
    app.include_router(module_router_module.router)
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def router_dummy_logger(monkeypatch, module_router_module):
    class DummyLogger:
        def __init__(self):
            self.errors = []
        def error(self, msg):
            self.errors.append(str(msg))
    dummy = DummyLogger()
    monkeypatch.setattr(module_router_module, "system_log", dummy)
    return dummy


def router_patch_async(monkeypatch, target, attr_name, *, result=None, exc: Exception = None):
    if exc is not None:
        async def _f(*args, **kwargs):
            raise exc
    else:
        async def _f(*args, **kwargs):
            return result
    monkeypatch.setattr(target, attr_name, _f)


def test_start_simulation_success_returns_status(router_test_client, monkeypatch, module_router_module, router_dummy_logger):
    # Mock scheduler behavior
    status = {"running": True, "started_at": "2025-01-01T00:00:00Z"}
    router_patch_async(monkeypatch, module_router_module.scheduler_service, "start_simulation", result=True)
    monkeypatch.setattr(module_router_module.scheduler_service, "get_simulation_status", lambda: status)

    resp = router_test_client.post("/simulator/start")
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("application/json")
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "시뮬레이션이 성공적으로 시작되었습니다."
    assert body["data"] == status
    assert router_dummy_logger.errors == []


def test_start_simulation_failure_returns_expected_message(router_test_client, monkeypatch, module_router_module, router_dummy_logger):
    router_patch_async(monkeypatch, module_router_module.scheduler_service, "start_simulation", result=False)

    resp = router_test_client.post("/simulator/start")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["message"] == "시뮬레이션 시작에 실패했습니다. (이미 실행 중이거나 API 서버 연결 실패)"
    assert body["data"] == {}
    # No errors should be logged on this code path
    assert router_dummy_logger.errors == []


def test_start_simulation_exception_results_in_500_and_logs(router_test_client, monkeypatch, module_router_module, router_dummy_logger):
    router_patch_async(monkeypatch, module_router_module.scheduler_service, "start_simulation", exc=Exception("boom-start"))

    resp = router_test_client.post("/simulator/start")
    assert resp.status_code == 500
    body = resp.json()
    assert "detail" in body
    assert body["detail"].startswith("내부 서버 오류:")
    assert "boom-start" in body["detail"]
    assert any("시뮬레이션 시작 API 오류" in msg for msg in router_dummy_logger.errors)


def test_stop_simulation_success(router_test_client, monkeypatch, module_router_module, router_dummy_logger):
    router_patch_async(monkeypatch, module_router_module.scheduler_service, "stop_simulation", result=True)

    resp = router_test_client.post("/simulator/stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "시뮬레이션이 성공적으로 종료되었습니다."
    assert body["data"] == {}
    assert router_dummy_logger.errors == []


def test_stop_simulation_failure(router_test_client, monkeypatch, module_router_module, router_dummy_logger):
    router_patch_async(monkeypatch, module_router_module.scheduler_service, "stop_simulation", result=False)

    resp = router_test_client.post("/simulator/stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["message"] == "시뮬레이션 종료에 실패했습니다. (실행 중이 아님)"
    assert body["data"] == {}
    assert router_dummy_logger.errors == []


def test_stop_simulation_exception_results_in_500_and_logs(router_test_client, monkeypatch, module_router_module, router_dummy_logger):
    router_patch_async(monkeypatch, module_router_module.scheduler_service, "stop_simulation", exc=Exception("boom-stop"))

    resp = router_test_client.post("/simulator/stop")
    assert resp.status_code == 500
    body = resp.json()
    assert "detail" in body
    assert body["detail"].startswith("내부 서버 오류:")
    assert "boom-stop" in body["detail"]
    assert any("시뮬레이션 종료 API 오류" in msg for msg in router_dummy_logger.errors)


def test_get_simulation_status_success(router_test_client, monkeypatch, module_router_module, router_dummy_logger):
    status = {"running": False, "last_stopped_at": "2025-02-02T12:34:56Z"}
    monkeypatch.setattr(module_router_module.scheduler_service, "get_simulation_status", lambda: status)

    resp = router_test_client.get("/simulator/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "시뮬레이션 상태를 성공적으로 조회했습니다."
    assert body["data"] == status
    assert router_dummy_logger.errors == []


def test_get_simulation_status_exception_results_in_500_and_logs(router_test_client, monkeypatch, module_router_module, router_dummy_logger):
    def _raise():
        raise Exception("boom-status")
    monkeypatch.setattr(module_router_module.scheduler_service, "get_simulation_status", _raise)

    resp = router_test_client.get("/simulator/status")
    assert resp.status_code == 500
    body = resp.json()
    assert "detail" in body
    assert body["detail"].startswith("내부 서버 오류:")
    assert "boom-status" in body["detail"]
    assert any("시뮬레이션 상태 조회 API 오류" in msg for msg in router_dummy_logger.errors)
