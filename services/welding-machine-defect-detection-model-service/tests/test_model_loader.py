from app.models.model_loader import load_scaler, load_model_file, load_threshold
import numpy as np


def test_vibration_model_loading(signal_type="vib"):
    # 테스트 1: scaler 로딩
    scaler = load_scaler(signal_type)
    sample_data = np.random.rand(1, 512)
    scaled_data = scaler.transform(sample_data)
    print("[✓] Scaler loaded and transform success:", scaled_data)

    # 테스트 2: 모델 로딩
    model = load_model_file(signal_type)
    print("[✓] Model loaded successfully:", model)

    # 테스트 3: threshold 로딩
    threshold = load_threshold(signal_type)
    print("[✓] Threshold loaded:", threshold)


def test_current_model_loading(signal_type="cur"):
    # 테스트 1: scaler 로딩
    scaler = load_scaler(signal_type)
    sample_data = np.random.rand(1, 1024)
    scaled_data = scaler.transform(sample_data)
    print("[✓] Scaler loaded and transform success:", scaled_data)

    # 테스트 2: 모델 로딩
    model = load_model_file(signal_type)
    print("[✓] Model loaded successfully:", model)

    # 테스트 3: threshold 로딩
    threshold = load_threshold(signal_type)
    print("[✓] Threshold loaded:", threshold)


if __name__ == "__main__":
    test_vibration_model_loading()
    test_current_model_loading()
