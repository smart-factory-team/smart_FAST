import joblib
import json
import numpy as np
from keras.models import load_model


def load_scaler(signal_type: str):
    try:
        return joblib.load(f"app/models/scalers/welding_scaler_{signal_type}.pkl")
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file for signal type '{signal_type}' not found")
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler for signal type '{signal_type}': {str(e)}")


def load_model_file(signal_type: str):
    return load_model(f"app/models/trained_models/welding_{signal_type}_model.keras")


def load_threshold(signal_type: str):
    with open("static/thresholds/welding_thresholds.json", "r") as f:
        return json.load(f)[signal_type]
