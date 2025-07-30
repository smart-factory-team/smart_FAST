from app.models.model_loader import load_scaler, load_model_file, load_threshold
import numpy as np
import pytest


@pytest.mark.parametrize("signal_type,features", [
    ("vib", 512),
    ("cur", 1024)
])
def test_model_loading(signal_type, features):
    # 테스트 1: scaler 로딩
    scaler = load_scaler(signal_type)
    assert scaler is not None, f"Scaler for {signal_type} should be loaded successfully"

    sample_data = np.random.rand(1, features)
    scaled_data = scaler.transform(sample_data)
    assert scaled_data.shape == sample_data.shape, "Scaled data should maintain input shape"

    # 테스트 2: 모델 로딩
    model = load_model_file(signal_type)
    assert model is not None, f"Model for {signal_type} should be loaded successfully"
    assert hasattr(model, 'predict'), "Model should have predict method"

    # 테스트 3: threshold 로딩
    threshold = load_threshold(signal_type)
    assert isinstance(threshold, (int, float)), "Threshold should be numeric"
    assert threshold > 0, "Threshold should be positive"


if __name__ == "__main__":
    pytest.main([__file__])
