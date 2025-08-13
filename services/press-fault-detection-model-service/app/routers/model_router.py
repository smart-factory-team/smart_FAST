from fastapi import APIRouter, HTTPException
from typing import Optional

from app.core.model_cache import model_cache

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info")
async def get_model_info():
    """
    모델의 정보 및 상태를 조회합니다.
    """
    try:
        model = model_cache.get("model")
        scaler = model_cache.get("scaler")
        threshold = model_cache.get("threshold")

        # 모델 로딩 상태 확인
        model_loaded = model is not None
        scaler_loaded = scaler is not None
        threshold_loaded = threshold is not None

        model_info = {
            "service_name": "프레스 설비(유압 펌프) 고장 예측 모델",
            "model_type": "Autoencoder",
            "organization": "23smartfactory",
            "repository": "press_fault_prediction",
            "model_files": {
                "model": "press_fault_detection.keras",
                "scaler": "press_fault_scaler.pkl",
                "threshold": "press_threshold.npy",
            },
            "status": {
                "model_loaded": model_loaded,
                "scaler_loaded": scaler_loaded,
                "threshold_loaded": threshold_loaded,
                "all_components_ready": all(
                    [model_loaded, scaler_loaded, threshold_loaded]
                ),
            },
            "input_features": ["AI0_Vibration", "AI1_Vibration", "AI2_Current"],
            "sequence_length": 20,
        }

        # 모델이 로드된 경우 추가 정보 제공
        if model_loaded and hasattr(model, "get_config"):
            try:
                model_config = model.get_config()
                model_info["model_details"] = {
                    "layers": len(model_config.get("layers", [])),
                    "input_shape": (
                        str(model.input_shape)
                        if hasattr(model, "input_shape")
                        else None
                    ),
                    "output_shape": (
                        str(model.output_shape)
                        if hasattr(model, "output_shape")
                        else None
                    ),
                }
            except Exception:
                model_info["model_details"] = "모델 세부 정보를 가져올 수 없습니다."

        # 임계값 정보
        if threshold_loaded:
            try:
                model_info["threshold_value"] = (
                    float(threshold) if threshold is not None else None
                )
            except Exception:
                model_info["threshold_value"] = "임계값을 읽을 수 없습니다."

        return model_info

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"모델 정보 조회 중 오류 발생: {str(e)}"
        ) from e
