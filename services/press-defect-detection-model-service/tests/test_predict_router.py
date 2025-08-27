import io
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Testing library and framework used: pytest + FastAPI's TestClient
from services.press_defect_detection_model_service.app.predict_router_module import (
    router,
    set_inference_service,
)

class DummyYoloModel:
    def __init__(self, loaded: bool = True) -> None:
        self.model_loaded = loaded

class DummyInferenceService:
    def __init__(self, single_result: Dict[str, Any] = None):
        self._single_result = single_result or {"success": True, "data": {"holes": []}}
        self._multi_result = {"success": True, "data": {"results": []}}
        self._inspection_result = {"success": True, "decision": "Pass"}
        self.yolo_model = DummyYoloModel(True)

    def predict_single_image(self, image_input: str, input_type: str) -> Dict[str, Any]:
        # Echo inputs to ensure we can assert on them
        result = dict(self._single_result)
        result["echo"] = {"image_input": image_input, "input_type": input_type}
        return result

    def predict_multiple_images(self, images: List[Dict[str, str]]) -> Dict[str, Any]:
        res = dict(self._multi_result)
        res["echo"] = {"images": images}
        return res

    def predict_inspection_batch(self, inspection_id: str, images: List[Dict[str, str]]) -> Dict[str, Any]:
        res = dict(self._inspection_result)
        res["echo"] = {"inspection_id": inspection_id, "images": images}
        return res

    def get_service_info(self) -> Dict[str, Any]:
        return {"name": "dummy", "version": "1.0.0"}

@pytest.fixture(autouse=True)
def app_and_client():
    # Fresh app per test to avoid cross-test contamination
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    # Reset inference service to None by setting a falsy stub where needed
    set_inference_service(None)  # type: ignore
    yield app, client
    # Teardown
    set_inference_service(None)  # type: ignore

def _valid_base64(length: int = 120) -> str:
    return "a" * length

def test_single_image_503_when_service_not_ready(app_and_client):
    _, client = app_and_client
    resp = client.post("/predict/", json={"image_base64": _valid_base64(), "image_name": "img"})
    assert resp.status_code == 503
    assert "추론 서비스가 준비되지 않았습니다." in resp.json().get("detail", "")

def test_multi_images_503_when_service_not_ready(app_and_client):
    _, client = app_and_client
    resp = client.post("/predict/multi", json={"images": [{"image": _valid_base64(), "name": "a"}]})
    assert resp.status_code == 503

def test_inspection_batch_503_when_service_not_ready(app_and_client):
    _, client = app_and_client
    resp = client.post("/predict/inspection", json={"inspection_id": "insp1", "images": [{"image": _valid_base64()}]*21})
    assert resp.status_code == 503

def test_file_upload_503_when_service_not_ready(app_and_client):
    _, client = app_and_client
    file_content = b"binaryimage"
    files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
    resp = client.post("/predict/file", files=files)
    assert resp.status_code == 503

def test_single_image_validation_422_for_short_base64(app_and_client):
    _, client = app_and_client
    resp = client.post("/predict/", json={"image_base64": "short", "image_name": "x"})
    assert resp.status_code == 422
    # Pydantic error message about validator
    assert any("유효하지 않은 base64 이미지" in (err.get("msg") or "") for err in resp.json().get("detail", []))

def test_multi_images_validation_0_images_422(app_and_client):
    _, client = app_and_client
    resp = client.post("/predict/multi", json={"images": []})
    assert resp.status_code == 422
    assert any("최소 1개의 이미지" in (err.get("msg") or "") for err in resp.json().get("detail", []))

def test_multi_images_validation_over_50_422(app_and_client):
    _, client = app_and_client
    over = [{"image": _valid_base64()} for _ in range(51)]
    resp = client.post("/predict/multi", json={"images": over})
    assert resp.status_code == 422
    assert any("최대 50장까지만 처리" in (err.get("msg") or "") for err in resp.json().get("detail", []))

def test_inspection_batch_validation_empty_id_422(app_and_client):
    _, client = app_and_client
    payload = {"inspection_id": "   ", "images": [{"image": _valid_base64()} for _ in range(21)]}
    resp = client.post("/predict/inspection", json=payload)
    assert resp.status_code == 422
    assert any("inspection_id는 필수" in (err.get("msg") or "") for err in resp.json().get("detail", []))

def test_single_image_success_happy_path_calls_service_with_base64(app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService(single_result={"success": True, "result": {"ok": True}})
    set_inference_service(svc)  # type: ignore

    payload = {"image_base64": _valid_base64(), "image_name": "foo.jpg"}
    resp = client.post("/predict/", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["result"]["ok"] is True
    assert data["echo"]["input_type"] == "base64"
    assert data["echo"]["image_input"] == payload["image_base64"]

def test_single_image_failure_returns_400(app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService(single_result={"success": False, "error": "bad image"})
    set_inference_service(svc)  # type: ignore

    resp = client.post("/predict/", json={"image_base64": _valid_base64()})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "bad image"

def test_single_image_exception_returns_500(monkeypatch, app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService()
    def boom(*args, **kwargs):
        raise RuntimeError("svc exploded")
    svc.predict_single_image = boom  # type: ignore
    set_inference_service(svc)  # type: ignore

    resp = client.post("/predict/", json={"image_base64": _valid_base64()})
    assert resp.status_code == 500
    assert "예측 중 오류 발생" in resp.json()["detail"]

def test_multi_images_success_builds_inputs_correctly(app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService()
    set_inference_service(svc)  # type: ignore

    imgs = [
        {"image": _valid_base64(), "name": "first"},
        {"image": _valid_base64(), "name": "second"},
        {"image": _valid_base64()},  # no name -> fallback
    ]
    resp = client.post("/predict/multi", json={"images": imgs})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    echoed = data["echo"]["images"]
    assert echoed[0]["type"] == "base64" and echoed[0]["name"] == "first"
    assert echoed[1]["type"] == "base64" and echoed[1]["name"] == "second"
    assert echoed[2]["type"] == "base64" and echoed[2]["name"] == "image_2"

def test_multi_images_failure_400(app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService()
    def fail(images):
        return {"success": False, "error": "multi fail"}
    svc.predict_multiple_images = fail  # type: ignore
    set_inference_service(svc)  # type: ignore

    imgs = [{"image": _valid_base64()}]
    resp = client.post("/predict/multi", json={"images": imgs})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "multi fail"

def test_multi_images_exception_500(app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService()
    def boom(images):
        raise ValueError("nope")
    svc.predict_multiple_images = boom  # type: ignore
    set_inference_service(svc)  # type: ignore

    resp = client.post("/predict/multi", json={"images": [{"image": _valid_base64()}]})
    assert resp.status_code == 500
    assert "예측 중 오류 발생" in resp.json()["detail"]

def test_inspection_batch_success_includes_id_and_names(app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService()
    set_inference_service(svc)  # type: ignore

    imgs = [{"image": _valid_base64()} for _ in range(3)]  # not 21 -> still allowed
    resp = client.post("/predict/inspection", json={"inspection_id": "insp-123", "images": imgs})
    assert resp.status_code == 200
    data = resp.json()
    echoed = data["echo"]
    assert echoed["inspection_id"] == "insp-123"
    assert echoed["images"][0]["name"] == "cam_0"
    assert echoed["images"][1]["name"] == "cam_1"

def test_inspection_batch_failure_400(app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService()
    def fail(inspection_id, images):
        return {"success": False, "error": "inspection fail"}
    svc.predict_inspection_batch = fail  # type: ignore
    set_inference_service(svc)  # type: ignore

    resp = client.post("/predict/inspection", json={"inspection_id": "id", "images": [{"image": _valid_base64()}]})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "inspection fail"

def test_inspection_batch_exception_500(app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService()
    def boom(inspection_id, images):
        raise RuntimeError("kaboom")
    svc.predict_inspection_batch = boom  # type: ignore
    set_inference_service(svc)  # type: ignore

    resp = client.post("/predict/inspection", json={"inspection_id": "id", "images": [{"image": _valid_base64()}]})
    assert resp.status_code == 500
    assert "예측 중 오류 발생" in resp.json()["detail"]

def test_file_upload_rejects_unsupported_extension(app_and_client):
    _, client = app_and_client
    set_inference_service(DummyInferenceService())  # type: ignore

    files = {"file": ("bad.txt", io.BytesIO(b"abc"), "text/plain")}
    resp = client.post("/predict/file", files=files)
    assert resp.status_code == 400
    assert "지원하지 않는 파일 형식" in resp.json()["detail"]

def test_file_upload_success_calls_predict_with_file_path_and_cleans_up(monkeypatch, app_and_client):
    _, client = app_and_client

    # Stub NamedTemporaryFile to a deterministic path and object
    tmp_called = {"path": "/tmp/test_upload.jpg", "written": b"", "closed": False}
    class FakeTmp:
        def __init__(self, delete=False, suffix=""):
            self.name = tmp_called["path"]
        def write(self, b: bytes):
            tmp_called["written"] += b
        def close(self):
            tmp_called["closed"] = True

    unlink_calls = []
    def fake_unlink(path: str):
        unlink_calls.append(path)

    svc = DummyInferenceService(single_result={"success": True, "ok": 1})
    set_inference_service(svc)  # type: ignore

    monkeypatch.setattr("services.press_defect_detection_model_service.app.predict_router_module.tempfile.NamedTemporaryFile", FakeTmp)
    monkeypatch.setattr("services.press_defect_detection_model_service.app.predict_router_module.os.unlink", fake_unlink)
    monkeypatch.setattr("services.press_defect_detection_model_service.app.predict_router_module.os.path.exists", lambda p: True)

    file_bytes = b"imagebytes"
    files = {"file": ("pic.jpg", io.BytesIO(file_bytes), "image/jpeg")}
    resp = client.post("/predict/file", files=files)
    assert resp.status_code == 200
    data = resp.json()
    # File info returned
    assert data["file_info"]["filename"] == "pic.jpg"
    assert data["file_info"]["file_size"] == len(file_bytes)
    assert data["file_info"]["content_type"] == "image/jpeg"
    # ensure temp wrote content and was closed
    assert tmp_called["written"] == file_bytes
    assert tmp_called["closed"] is True
    # ensure unlink called for cleanup
    assert unlink_calls == [tmp_called["path"]]
    # ensure service received file_path input_type
    assert data["echo"]["input_type"] == "file_path"
    assert data["echo"]["image_input"] == tmp_called["path"]

def test_file_upload_service_failure_returns_400(monkeypatch, app_and_client):
    _, client = app_and_client

    class FakeTmp:
        def __init__(self, delete=False, suffix=""):
            self.name = "/tmp/fail.jpg"
        def write(self, b: bytes): pass
        def close(self): pass

    svc = DummyInferenceService(single_result={"success": False, "error": "bad file"})
    set_inference_service(svc)  # type: ignore

    monkeypatch.setattr("services.press_defect_detection_model_service.app.predict_router_module.tempfile.NamedTemporaryFile", FakeTmp)
    monkeypatch.setattr("services.press_defect_detection_model_service.app.predict_router_module.os.unlink", lambda p: None)
    monkeypatch.setattr("services.press_defect_detection_model_service.app.predict_router_module.os.path.exists", lambda p: True)

    files = {"file": ("pic.jpg", io.BytesIO(b"img"), "image/jpeg")}
    resp = client.post("/predict/file", files=files)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "bad file"

def test_service_info_and_health_when_ready(app_and_client):
    _, client = app_and_client
    svc = DummyInferenceService()
    set_inference_service(svc)  # type: ignore

    # service-info
    resp = client.get("/predict/service-info")
    assert resp.status_code == 200
    assert resp.json()["name"] == "dummy"

    # health
    resp = client.get("/predict/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "prediction"
    assert isinstance(data["timestamp"], str)
    assert data["model_loaded"] is True