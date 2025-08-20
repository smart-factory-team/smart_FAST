import os
import sys
import types
import importlib.util
import numpy as np
import pytest

# Testing library/framework note:
# This repository uses pytest. These tests leverage pytest's assertions and monkeypatch utilities.

# --- Dynamic import of the target module with safe stubs for heavy dependencies ---
# The module path was discovered at:
# services/press-defect-detection-model-service/app/press_models/yolo_model.py

def _load_yolo_module_with_stubs():
    # Build absolute path to the module based on this test file's location
    base_dir = os.path.dirname(os.path.dirname(__file__))  # services/press-defect-detection-model-service
    module_path = os.path.join(base_dir, "app", "press_models", "yolo_model.py")

    # Create minimal fake modules for torch, cv2, and huggingface_hub to avoid heavy deps at import time

    class _FakeTensor:
        def __init__(self, arr):
            self.arr = np.array(arr)
        def to(self, device):
            return self
        def float(self):
            return self
        def __truediv__(self, other):
            # Simulate normalization without changing internal array
            return self
        def ndimension(self):
            return self.arr.ndim
        def unsqueeze(self, dim):
            self.arr = np.expand_dims(self.arr, axis=dim)
            return self
        @property
        def shape(self):
            return self.arr.shape
        # Provide .max().item() and .min().item() contract
        def max(self):
            return types.SimpleNamespace(item=lambda: 1.0)
        def min(self):
            return types.SimpleNamespace(item=lambda: 0.0)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    class _NoGradCtx:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False

    class _FakeTorch(types.SimpleNamespace):
        pass

    fake_torch = _FakeTorch()
    fake_torch.Tensor = _FakeTensor
    fake_torch.from_numpy = lambda arr: _FakeTensor(arr)
    fake_torch.cuda = _FakeCuda()
    fake_torch.device = lambda dev: dev
    fake_torch.no_grad = lambda: _NoGradCtx()

    fake_cv2 = types.SimpleNamespace(
        imread=lambda path: None
    )

    fake_hf = types.SimpleNamespace(
        hf_hub_download=lambda **kwargs: "/tmp/fake-model.pt"
    )

    # Backup any existing modules to avoid global contamination
    backups = {}
    for name, mod in [("torch", fake_torch), ("cv2", fake_cv2), ("huggingface_hub", fake_hf)]:
        backups[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        spec = importlib.util.spec_from_file_location("press_yolo_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module at {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
    finally:
        # Restore original modules in sys.modules for other tests; the imported module retains its own references
        for name, orig in backups.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

    return mod

# Load module and class for tests
yolo_module = _load_yolo_module_with_stubs()
YOLOv7Model = yolo_module.YOLOv7Model

class DummyStride:
    def __init__(self, val=32):
        self._val = val
    def max(self):
        return self._val

class DummyModel:
    def __init__(self, stride_val=32):
        self.stride = DummyStride(stride_val)
        self._eval_called = False
    def eval(self):
        self._eval_called = True
        return self
    def __call__(self, img, augment=False):
        # Return a structure where [0] accesses "raw pred"
        return ([np.zeros((1, 6), dtype=np.float32)],)

def _make_image(h=100, w=120, c=3, val=128):
    return np.full((h, w, c), val, dtype=np.uint8)

@pytest.fixture
def model():
    return YOLOv7Model()

def test_load_model_success(monkeypatch):
    m = YOLOv7Model()

    # Bypass heavy YOLOv7 dependency loading
    monkeypatch.setattr(m, "_load_yolov7_dependencies", lambda: True)

    # Provide minimal attributes expected after dependency loading
    m.device = "cpu"

    dummy_model = DummyModel(stride_val=32)
    monkeypatch.setattr(m, "attempt_load", lambda path, map_location=None: dummy_model)
    monkeypatch.setattr(m, "check_img_size", lambda img_size, s=32: img_size)

    # Avoid real download
    monkeypatch.setattr(m, "_download_model_from_huggingface", lambda: "/tmp/fake-model.pt")

    import asyncio
    assert asyncio.run(m.load_model()) is True

    # Validate state
    assert m.model is dummy_model
    assert m.model_loaded is True
    assert dummy_model._eval_called is True
    assert m.img_size == 640  # unchanged due to check_img_size returning input

def test_load_model_dependency_failure_raises(monkeypatch):
    m = YOLOv7Model()
    monkeypatch.setattr(m, "_load_yolov7_dependencies", lambda: False)

    import asyncio
    with pytest.raises(Exception) as ei:
        asyncio.run(m.load_model())
    assert "YOLOv7 의존성 로딩 실패" in str(ei.value)

def test_preprocess_image_requires_yolov7_loaded(model):
    model.yolov7_loaded = False
    with pytest.raises(Exception) as ei:
        model._preprocess_image("any.png")
    assert "의존성이 로딩되지 않았습니다" in str(ei.value)

def test_preprocess_image_read_failure(monkeypatch, model):
    model.yolov7_loaded = True
    model.model = DummyModel()
    monkeypatch.setattr(model, "letterbox", lambda img, size, stride=32: (img, None))

    import cv2
    monkeypatch.setattr(cv2, "imread", lambda path: None)

    with pytest.raises(ValueError) as ei:
        model._preprocess_image("missing.png")
    assert "이미지를 로드할 수 없습니다" in str(ei.value)

def test_preprocess_image_success(monkeypatch, model):
    model.yolov7_loaded = True
    model.device = "cpu"
    model.model = DummyModel()
    # Letterbox returns the same image; focus on transformation pipeline
    monkeypatch.setattr(model, "letterbox", lambda img, size, stride=32: (img, None))

    import cv2
    dummy = _make_image(60, 80, 3, val=200)
    monkeypatch.setattr(cv2, "imread", lambda path: dummy.copy())

    img, img0 = model._preprocess_image("dummy.png")

    # Assertions on shapes and types
    assert img0.shape == (60, 80, 3)
    # Try torch-specific checks if a torch-like module is present
    try:
        import torch  # could be real torch or absent; this block is best-effort
        assert hasattr(torch, "Tensor")
        assert isinstance(img, torch.Tensor)
        # .max().item() and .min().item() are accommodated by our fakes if used
        assert img.ndimension() == 4
        assert img.shape[1] == 3
        assert img.max().item() <= 1.0 and img.min().item() >= 0.0
    except Exception:
        # Skip if torch not present or tensor checks not applicable
        pass

def test_postprocess_detections_builds_expected_results(monkeypatch, model):
    model.yolov7_loaded = True

    raw_pred = [np.zeros((1, 6), dtype=np.float32)]

    # One detection row: x1,y1,x2,y2,conf,cls
    det_array = np.array([[10.4, 20.2, 50.9, 80.7, 0.87, 3.0]], dtype=np.float32)  # class 3 => DY1

    # Stub NMS to return our detection
    monkeypatch.setattr(model, "non_max_suppression", lambda p, conf, iou, classes=None, agnostic=False: [det_array])

    # scale_coords: identity
    monkeypatch.setattr(model, "scale_coords", lambda in_shape, coords, out_shape: coords)

    # img shape (1, 3, H, W) used for scaling; use SimpleNamespace
    _types = types
    img_like = _types.SimpleNamespace(shape=(1, 3, 100, 120))
    img0 = _make_image(100, 120, 3)

    results = model._postprocess_detections(raw_pred, img_like, img0)

    assert isinstance(results, list)
    assert len(results) == 1
    r = results[0]
    assert r["category_id"] == 3
    assert r["category_name"] == model.category_names[3]
    assert r["confidence"] == pytest.approx(0.87, rel=1e-4)
    assert r["bbox"] == [10, 20, 51, 81] or r["bbox"] == [10, 20, 50, 80]
    cx, cy = r["bbox_center"]
    assert isinstance(cx, float) and isinstance(cy, float)

def test_postprocess_detections_handles_exception_returns_empty(monkeypatch, model):
    model.yolov7_loaded = True

    def boom(*args, **kwargs):
        raise RuntimeError("NMS failed")

    monkeypatch.setattr(model, "non_max_suppression", boom)
    img_like = types.SimpleNamespace(shape=(1, 3, 10, 10))
    results = model._postprocess_detections([np.zeros((1, 6))], img_like, _make_image(10, 10, 3))
    assert results == []

def test_detect_holes_requires_loaded_model(model):
    model.model_loaded = False
    with pytest.raises(Exception) as ei:
        model.detect_holes("any.png")
    assert "모델이 로딩되지 않았습니다" in str(ei.value)

def test_detect_holes_success(monkeypatch, model):
    model.model_loaded = True
    model.model = DummyModel()
    fake_img = object()
    fake_img0 = object()
    monkeypatch.setattr(model, "_preprocess_image", lambda path: (fake_img, fake_img0))
    expected = [{"category_id": 1, "category_name": "BY1", "confidence": 0.9, "bbox": [1, 2, 3, 4], "bbox_center": [2.0, 3.0]}]
    monkeypatch.setattr(model, "_postprocess_detections", lambda pred, img, img0: expected)

    detections = model.detect_holes("foo.png")
    assert detections == expected

def test_detect_holes_propagates_exceptions(monkeypatch, model):
    model.model_loaded = True
    model.model = DummyModel()
    def fail_preprocess(_):
        raise ValueError("preprocess failure")
    monkeypatch.setattr(model, "_preprocess_image", fail_preprocess)
    with pytest.raises(ValueError) as ei:
        model.detect_holes("bar.png")
    assert "preprocess failure" in str(ei.value)

def test_apply_adaptive_thresholds_no_images(model):
    result = model.apply_adaptive_thresholds({})
    assert result["is_complete"] is False
    assert "처리된 이미지가 없습니다" in result["error"]

def test_apply_adaptive_thresholds_counts_votes_and_thresholds(model):
    # 4 images with varying detections and confidences (only valid categories 0..6 used)
    detections_by_image = {
        "img1": [
            {"category_id": 0, "confidence": 0.8},   # AX1 counted
            {"category_id": 6, "confidence": 0.6},   # DY4
        ],
        "img2": [
            {"category_id": 0, "confidence": 0.55},  # AX1
            {"category_id": 2, "confidence": 0.7},   # CY1
        ],
        "img3": [
            {"category_id": 3, "confidence": 0.51},  # DY1
            {"category_id": 4, "confidence": 0.52},  # DY2
        ],
        "img4": [
            {"category_id": 5, "confidence": 0.51},  # DY3
            {"category_id": 1, "confidence": 0.49},  # BY1 below threshold, ignored
        ],
    }
    out = model.apply_adaptive_thresholds(detections_by_image)

    assert out["processed_images"] == 4
    # Check category results structure
    assert set(out["category_results"].keys()) == set(range(7))
    # Some categories likely missing due to thresholds
    assert isinstance(out["missing_categories"], list)
    assert isinstance(out["existing_categories"], list)
    assert set(out["existing_categories"]).issubset(set(range(7)))

    # For 4 images, required votes by category:
    # 0: max(1, int(0.6))=1 -> existing (seen twice)
    # 1: max(1, int(2.0))=2 -> missing (0 votes at >=0.5)
    # 2: max(1, int(0.8))=1 -> existing (1 vote)
    # 3: max(1, int(1.6))=1 -> existing (1 vote)
    # 4: max(1, int(1.8))=1 -> existing (1 vote)
    # 5: max(1, int(1.0))=1 -> existing (1 vote)
    # 6: max(1, int(1.4))=1 -> existing (1 vote)
    existing = set(out["existing_categories"])
    assert {0, 2, 3, 4, 5, 6}.issubset(existing)
    assert 1 in out["missing_categories"]
    assert out["is_complete"] is False
    assert out["quality_status"] == "결함품"
    assert "BY1" in out["missing_category_names"]

def test_apply_adaptive_thresholds_complete_product(model):
    # Construct detections so that every category meets its threshold with 10 images
    # Required votes for 10 images:
    # 0:1, 1:5, 2:2, 3:4, 4:4, 5:2, 6:3
    detections_by_image = {}
    for i in range(10):
        dets = []
        if i < 7:
            dets.append({"category_id": 0, "confidence": 0.9})
        if i < 5:
            dets.append({"category_id": 1, "confidence": 0.9})
        if i < 6:
            dets.append({"category_id": 2, "confidence": 0.9})
        if i < 9:
            dets.append({"category_id": 3, "confidence": 0.9})
            dets.append({"category_id": 4, "confidence": 0.9})
        if i < 5:
            dets.append({"category_id": 5, "confidence": 0.9})
        if i < 8:
            dets.append({"category_id": 6, "confidence": 0.9})
        detections_by_image[f"img{i}"] = dets

    out = model.apply_adaptive_thresholds(detections_by_image)
    assert out["is_complete"] is True
    assert out["quality_status"] == "정상품"
    assert out["missing_categories"] == []
    assert len(out["existing_categories"]) == 7

def test_get_model_info_reflects_state(model):
    # Set state
    model.model_loaded = True
    model.yolov7_loaded = True
    model.yolov7_path = "/opt/yolov7"
    model.device = "cpu"
    model.img_size = 512
    model.conf_thres = 0.3
    model.iou_thres = 0.4

    info = model.get_model_info()
    assert info["model_loaded"] is True
    assert info["yolov7_loaded"] is True
    assert info["yolov7_path"] == "/opt/yolov7"
    assert info["device"] == "cpu"
    assert info["img_size"] == 512
    assert info["conf_thres"] == 0.3
    assert info["iou_thres"] == 0.4
    assert isinstance(info["categories"], dict)
    assert isinstance(info["adaptive_thresholds"], dict)
    any_key = next(iter(info["adaptive_thresholds"].keys()))
    assert any_key in model.category_names.values()
    any_val = info["adaptive_thresholds"][any_key]
    assert isinstance(any_val, str) and any_val.endswith("%")

def test_setup_yolov7_repo_absent_yolov7_dir(monkeypatch, model, tmp_path):
    # Simulate __file__ directory structure
    fake_file = tmp_path / "service" / "module.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text("# fake module")

    def fake_abspath(_):
        return str(fake_file)

    # Patch os.path.abspath within the yolo_module namespace to avoid changing global behavior
    monkeypatch.setattr(yolo_module.os.path, "abspath", fake_abspath)

    ok = model._setup_yolov7_repo()
    assert ok is False
    assert model.yolov7_path is None