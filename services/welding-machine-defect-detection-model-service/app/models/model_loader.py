import joblib
import json
import numpy as np
from keras.models import load_model


def load_scaler(signal_type: str):
    try:
        return joblib.load(f"app/models/scalers/welding_scaler_{signal_type}.pkl")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaler file for signal type '{signal_type}' not found")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load scaler for signal type '{signal_type}': {str(e)}")


def load_model_file(signal_type: str):
    try:
        return load_model(f"app/models/trained_models/welding_{signal_type}_model.keras")
    except (FileNotFoundError, OSError) as e:
        raise FileNotFoundError(
            f"Model file for signal type '{signal_type}' not found: {e}")


def load_threshold(signal_type: str):
    try:
        with open("static/thresholds/welding_thresholds.json", "r") as f:
            thresholds = json.load(f)
            if signal_type not in thresholds:
                raise KeyError(
                    f"Threshold for signal type '{signal_type}' not found in configuration")
            return thresholds[signal_type]
    except FileNotFoundError:
        raise FileNotFoundError("Threshold configuration file not found")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in threshold configuration file: {str(e)}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load threshold for signal type '{signal_type}': {str(e)}")
