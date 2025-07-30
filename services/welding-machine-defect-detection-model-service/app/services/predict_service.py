import numpy as np
from app.models.model_loader import load_model_file, load_scaler, load_threshold


def predict_anomaly(signal_type: str, values: list):
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
