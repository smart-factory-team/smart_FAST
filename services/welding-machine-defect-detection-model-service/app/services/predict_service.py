import numpy as np
from app.models.model_loader import load_model_file, load_scaler, load_threshold


def predict_anomaly(signal_type: str, values: list):
    check_input_validation(signal_type, values)

    arr = np.array(values).reshape(1, -1)

    scaler = load_scaler(signal_type)
    model = load_model_file(signal_type)
    threshold = load_threshold(signal_type)

    scaled = scaler.transform(arr)
    reshaped = scaled.reshape(scaled.shape[0], scaled.shape[1], 1)

    pred = model.predict(reshaped)
    mae = np.mean(np.abs(pred - reshaped))

    return {
        "signal_type": signal_type,
        "mae": float(mae),
        "threshold": threshold,
        "status": "anomaly" if mae > threshold else "normal"
    }


def check_input_validation(signal_type: str, values: list):
    if not isinstance(signal_type, str) or not signal_type.strip():
        raise ValueError("signal_type must be a non-empty string")

    if signal_type not in ["cur", "vib"]:
        raise ValueError("signal_type must be either 'cur' or 'vib'")

    if not isinstance(values, list) or len(values) == 0:
        raise ValueError("values must be a non-empty list")

    if not all(isinstance(v, (int, float)) for v in values):
        raise ValueError("all values must be numeric")

    expected_length = 1024 if signal_type == "cur" else 512
    if len(values) != expected_length:
        raise ValueError(
            f"values must have exactly {expected_length} elements for signal_type '{signal_type}'")
