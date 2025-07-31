import json
import numpy as np
import joblib
from keras.models import load_model
from huggingface_hub import hf_hub_download

ORG = "23smartfactory"
REPO = "Welding_model"


def load_scaler(signal_type: str):
    try:
        filename = f"welding_scaler_{signal_type}.pkl"
        scaler_path = hf_hub_download(
            repo_id=f"{ORG}/{REPO}", filename=filename)
        return joblib.load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaler file '{filename}' not found on Hugging Face")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load scaler from Hugging Face: {str(e)}")


def load_model_file(signal_type: str):
    try:
        filename = f"welding_{signal_type}_model.keras"
        model_path = hf_hub_download(
            repo_id=f"{ORG}/{REPO}", filename=filename)
        return load_model(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file '{filename}' not found on Hugging Face")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from Hugging Face: {str(e)}")


def load_threshold(signal_type: str):
    try:
        filename = "welding_thresholds.json"
        json_path = hf_hub_download(repo_id=f"{ORG}/{REPO}", filename=filename)
        with open(json_path, "r") as f:
            thresholds = json.load(f)
        if signal_type not in thresholds:
            raise KeyError(
                f"Threshold for signal type '{signal_type}' not found")
        return thresholds[signal_type]
    except FileNotFoundError:
        raise FileNotFoundError(
            "Threshold JSON file not found on Hugging Face")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in threshold file: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load threshold: {str(e)}")
