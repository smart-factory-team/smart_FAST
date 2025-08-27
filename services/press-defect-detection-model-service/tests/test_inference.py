# This file was created to host tests for InferenceService. If the implementation
# is located elsewhere, adjust imports accordingly. For now, tests assume the
# InferenceService class is defined in this module.
#
# =========================
# Pytest unit tests section
# =========================
import base64 as _t_base64
import io as _t_io
import os as _t_os
from datetime import datetime as _t_datetime

import pytest

from press_defect_detection_model_service.inference_service import InferenceService

try:
    from PIL import Image as _t_Image
except Exception:  # pragma: no cover - PIL should already be a dependency in this service
    _t_Image = None


def _generate_test_base64_image(with_header: bool = True) -> str:
    """
    Helper to generate a tiny in-memory JPEG image encoded as base64.
    Requires Pillow; if not available, raises RuntimeError for clear feedback.
    """
    if _t_Image is None:
        raise RuntimeError("Pillow (PIL) is required for image-based tests.")
    img = _t_Image.new("RGB", (2, 2), color=(123, 234, 45))
    buf = _t_io.BytesIO()
    img.save(buf, format="JPEG")
    raw = buf.getvalue()
    b64 = _t_base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}" if with_header else b64


class _MockYOLO:
    def __init__(self, detections=None, adaptive=None, model_info=None, throw_on_detect: bool=False, throw_on_thresholds: bool=False):
        self._detections = detections if detections is not None else []
        self._adaptive = adaptive if adaptive is not None else {"quality_status": "Unknown", "is_complete": False, "missing_category_names": []}
        self._model_info = model_info if model_info is not None else {"name": "MockYOLO", "version": "0.0.1"}
        self._throw_on_detect = throw_on_detect
        self._throw_on_thresholds = throw_on_thresholds

    def detect_holes(self, image_path: str):
        if self._throw_on_detect:
            raise RuntimeError("model failure")
        # ensure the path is a string
        assert isinstance(image_path, str)
        return self._detections

    def apply_adaptive_thresholds(self, detections_by_image: dict):
        if self._throw_on_thresholds:
            raise RuntimeError("thresholds failure")
        assert isinstance(detections_by_image, dict)
        return self._adaptive

    def get_model_info(self):
        return self._model_info


@pytest.fixture
def default_detections():
    # Prepare a set with duplicate categories to validate unique "categories_detected" and "category_names_detected"
    return [
        {"category_id": 1, "category_name": "Hole-A", "x": 10, "y": 20, "w": 5, "h": 5, "confidence": 0.95},
        {"category_id": 2, "category_name": "Hole-B", "x": 30, "y": 40, "w": 6, "h": 6, "confidence": 0.85},
        {"category_id": 1, "category_name": "Hole-A", "x": 50, "y": 60, "w": 4, "h": 4, "confidence": 0.90},
    ]


@pytest.fixture
def service_with_mock(default_detections):
    return InferenceService(_MockYOLO(detections=default_detections))


def test__save_base64_image_valid_and_cleanup():
    service = InferenceService(_MockYOLO())
    b64 = _generate_test_base64_image(with_header=True)
    path = service._save_base64_image(b64)
    try:
        assert _t_os.path.exists(path), "Temporary image file should exist after saving a valid base64 image"
        # Verify it is an actual JPEG readable by PIL
        if _t_Image is not None:
            with _t_Image.open(path) as img:
                assert img.format == "JPEG"
    finally:
        # Ensure cleanup works and does not raise
        service._cleanup_temp_file(path)
        assert not _t_os.path.exists(path), "Temporary image file should be removed by cleanup"


def test__save_base64_image_invalid_raises_value_error():
    service = InferenceService(_MockYOLO())
    with pytest.raises(ValueError):
        service._save_base64_image("not_base64_at_all")


def test__cleanup_temp_file_nonexistent_is_graceful(tmp_path):
    service = InferenceService(_MockYOLO())
    nonexistent = str(tmp_path / "nope.jpg")
    # Should not raise even if file doesn't exist
    service._cleanup_temp_file(nonexistent)
    # Nothing to assert other than no exception


def test_predict_single_image_with_base64_success(service_with_mock, default_detections, monkeypatch):
    """
    Validate:
    - success True
    - detections collected
    - categories_detected and category_names_detected are unique sets
    - temporary file is cleaned up via _cleanup_temp_file
    """
    # Prepare a real base64 so PIL runs, but intercept the cleanup to capture the path
    b64 = _generate_test_base64_image(with_header=False)

    cleaned = {"called": False, "path": None}

    orig_save = service_with_mock._save_base64_image
    def _spy_save(b64s):
        path = orig_save(b64s)
        # Ensure file exists when saved
        assert _t_os.path.exists(path)
        cleaned["path"] = path
        return path

    def _spy_cleanup(path):
        cleaned["called"] = True
        cleaned["path"] = path
        # Call original cleanup to actually delete file
        return InferenceService._cleanup_temp_file(service_with_mock, path)

    monkeypatch.setattr(service_with_mock, "_save_base64_image", _spy_save)
    monkeypatch.setattr(service_with_mock, "_cleanup_temp_file", _spy_cleanup)

    result = service_with_mock.predict_single_image(b64, input_type="base64")
    assert result["success"] is True
    assert result["image_info"]["input_type"] == "base64"
    assert result["image_info"]["processed"] is True
    assert result["detections"]["total_count"] == len(default_detections)
    assert result["detections"]["holes"] == default_detections
    # Unique categories
    assert sorted(result["categories_detected"]) == sorted(list({d["category_id"] for d in default_detections}))
    assert sorted(result["category_names_detected"]) == sorted(list({d["category_name"] for d in default_detections}))
    # Timestamp sanity
    assert isinstance(result["timestamp"], str) and len(result["timestamp"]) > 0

    # Cleanup verification
    assert cleaned["called"] is True
    assert cleaned["path"] is not None
    assert not _t_os.path.exists(cleaned["path"]), "Temp file should be deleted after prediction"


def test_predict_single_image_with_missing_file_path_returns_error():
    svc = InferenceService(_MockYOLO())
    result = svc.predict_single_image("does_not_exist.jpg", input_type="file_path")
    assert result["success"] is False
    assert "존재하지" in result["error"] or "exist" in result["error"]  # robust to locale/environment
    assert isinstance(result["timestamp"], str)


def test_predict_single_image_model_exception_returns_error(monkeypatch):
    svc = InferenceService(_MockYOLO(throw_on_detect=True))
    b64 = _generate_test_base64_image()
    result = svc.predict_single_image(b64, input_type="base64")
    assert result["success"] is False
    assert "model failure" in result["error"]
    assert isinstance(result["timestamp"], str)


def test_predict_multiple_images_mixed_outcomes(monkeypatch):
    """
    Validate aggregation:
    - processed_images count
    - failed_images count and details
    - detections_by_image keys use provided 'name' or default `image_{idx}`
    - adaptive thresholds are applied and returned
    """
    # Prepare a service whose predict_single_image we will stub
    adaptive = {"quality_status": "OK", "is_complete": True, "missing_category_names": []}
    svc = InferenceService(_MockYOLO(adaptive=adaptive))

    calls = {"count": 0}
    def _fake_single(image_data, input_type="file_path"):
        idx = calls["count"]
        calls["count"] += 1
        if idx == 0:
            # success with holes
            return {"success": True, "detections": {"holes": [{"category_id": 1, "category_name": "A"}]}}
        elif idx == 1:
            # failure
            return {"success": False, "error": "bad image"}
        else:
            # success empty detections
            return {"success": True, "detections": {"holes": []}}

    monkeypatch.setattr(svc, "predict_single_image", _fake_single)

    inputs = [
        {"image": "img1", "type": "base64", "name": "custom_name"},
        {"image": "img2", "type": "file_path"},
        {"image": "img3", "type": "base64"},  # no name provided -> defaults to image_2
    ]
    result = svc.predict_multiple_images(inputs)

    assert result["success"] is True
    assert result["processing_summary"]["total_images"] == 3
    assert result["processing_summary"]["processed_images"] == 2
    assert result["processing_summary"]["failed_images"] == 1
    assert len(result["processing_summary"]["failed_details"]) == 1
    assert result["processing_summary"]["failed_details"][0]["name"] in {"image_1", "img2", "custom_name", "image_0"}  # name derived; robust check
    # Detailed detections by provided/default name
    assert "custom_name" in result["detailed_detections"]
    assert "image_2" in result["detailed_detections"]
    assert result["quality_inspection"] == adaptive


def test_predict_multiple_images_thresholds_exception_returns_error(monkeypatch):
    svc = InferenceService(_MockYOLO(throw_on_thresholds=True))
    # Simpler: stub predict_single_image to always succeed once to reach thresholds
    monkeypatch.setattr(svc, "predict_single_image", lambda img, input_type="file_path": {"success": True, "detections": {"holes": []}})
    result = svc.predict_multiple_images([{"image": "x", "type": "base64"}])
    assert result["success"] is False
    assert "thresholds failure" in result["error"]


def test_predict_inspection_batch_complete_and_pass(monkeypatch):
    svc = InferenceService(_MockYOLO())
    # 21 images, choose quality is_complete True -> Pass
    images = [{"image": f"img{i}", "type": "base64"} for i in range(21)]
    fake_multi = {
        "success": True,
        "processing_summary": {"total_images": 21, "processed_images": 21, "failed_images": 0, "failed_details": []},
        "quality_inspection": {"quality_status": "Good", "is_complete": True, "missing_category_names": []},
        "detailed_detections": {},
        "timestamp": _t_datetime.now().isoformat(),
    }
    monkeypatch.setattr(svc, "predict_multiple_images", lambda imgs: fake_multi)

    result = svc.predict_inspection_batch("insp-123", images)
    assert result["success"] is True
    assert result["inspection_info"]["inspection_id"] == "insp-123"
    assert result["inspection_info"]["expected_images"] == 21
    assert result["inspection_info"]["actual_images"] == 21
    assert result["inspection_info"]["is_complete_dataset"] is True
    assert result["final_judgment"]["quality_status"] == "Good"
    assert result["final_judgment"]["is_complete"] is True
    assert result["final_judgment"]["recommendation"] == "Pass"


def test_predict_inspection_batch_incomplete_reject(monkeypatch):
    svc = InferenceService(_MockYOLO())
    images = [{"image": "img", "type": "base64"} for _ in range(20)]
    fake_multi = {
        "success": True,
        "processing_summary": {"total_images": 20, "processed_images": 19, "failed_images": 1, "failed_details": [{"name": "image_4", "error": "bad"}]},
        "quality_inspection": {"quality_status": "Poor", "is_complete": False, "missing_category_names": ["Hole-Z"]},
        "detailed_detections": {},
        "timestamp": _t_datetime.now().isoformat(),
    }
    monkeypatch.setattr(svc, "predict_multiple_images", lambda imgs: fake_multi)
    result = svc.predict_inspection_batch("insp-999", images)
    assert result["success"] is True
    assert result["inspection_info"]["inspection_id"] == "insp-999"
    assert result["inspection_info"]["expected_images"] == 21
    assert result["inspection_info"]["actual_images"] == 20
    assert result["inspection_info"]["is_complete_dataset"] is False
    assert result["final_judgment"]["quality_status"] == "Poor"
    assert result["final_judgment"]["is_complete"] is False
    assert result["final_judgment"]["recommendation"] == "Reject"


def test_predict_inspection_batch_failure_propagates(monkeypatch):
    svc = InferenceService(_MockYOLO())
    monkeypatch.setattr(svc, "predict_multiple_images", lambda imgs: {"success": False, "error": "aggregation failed", "timestamp": _t_datetime.now().isoformat()})
    result = svc.predict_inspection_batch("insp-err", [{"image": "x", "type": "file_path"}])
    assert result["success"] is False
    assert result["error"] == "aggregation failed"
    # When inner call fails, inspection_info/final_judgment are not present
    assert "inspection_info" not in result
    assert "final_judgment" not in result


def test_get_service_info_structure():
    model_info = {"name": "Mock-YOLO-v8", "version": "1.2.3", "extra": {"param": 1}}
    svc = InferenceService(_MockYOLO(model_info=model_info))
    info = svc.get_service_info()
    assert info["service_name"] == "Press Defect Detection Inference Service"
    assert info["model_info"] == model_info
    assert set(info["supported_formats"]) >= {"JPEG", "PNG", "JPG"}
    assert set(info["input_methods"]) >= {"file_upload", "base64"}
    assert set(info["features"]) >= {"single_image_detection", "multi_image_quality_inspection", "adaptive_threshold_evaluation", "batch_processing"}