from huggingface_hub import hf_hub_download
import numpy as np
import joblib
from keras.models import load_model as keras_load_model
import logging

logger = logging.getLogger(__name__)

ORG = "23smartfactory"
REPO = "press_fault_prediction"

def load_model():
    try:
        repo_id = f"{ORG}/{REPO}"
        model_filename = "press_fault_detection.keras"
        model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename
            )
        return keras_load_model(model_path)
        
    except Exception as e:
        logger.error(f"Keras 모델 로딩 실패: {str(e)}")
        return None
        
def load_scaler():
    try:
        repo_id = f"{ORG}/{REPO}"
        scaler_filename = "press_fault_scaler.pkl"
        scaler_path = hf_hub_download(
            repo_id=repo_id,
            filename=scaler_filename
        )
        return joblib.load(scaler_path)
        
    except Exception as e:
        logger.error(f"스케일러 로딩 실패: {str(e)}")
        return None
                
def load_threshold():
    try:
        repo_id = f"{ORG}/{REPO}"
        threshold_filename = "press_threshold.npy"
        threshold_path = hf_hub_download(
            repo_id=repo_id,
            filename=threshold_filename
        )
        return np.load(threshold_path)
        
    except Exception as e:
        logger.error(f"임계치 로딩 실패: {str(e)}")
        return None
    
def load_all():
    model = load_model()
    scaler = load_scaler()
    threshold = load_threshold()
    
    if model is None or scaler is None or threshold is None:
        missing_items = []
        if model is None:
            missing_items.append("model")
        if scaler is None:
            missing_items.append("scaler")
        if threshold is None:
            missing_items.append("threshold")
        
        raise RuntimeError(f"다음 artifacts 로딩 실패: {', '.join(missing_items)}")
    
    return {
        'model': model,
        'scaler': scaler,
        'threshold': threshold
    }