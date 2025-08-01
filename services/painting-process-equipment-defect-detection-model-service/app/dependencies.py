import joblib
import shap
import yaml
from fastapi import HTTPException
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download # huggingface_hub 라이브러리 임포트

# 전역 변수 초기화
_model: Optional[Any] = None
_config: Optional[Dict[str, Any]] = None
_explainer: Optional[Any] = None
_initialization_complete: bool = False  # 초기화 완료 상태를 추적하는 플래그


# 초기화 실패 시 전역 상태를 리셋하는 헬퍼 함수
def _reset_global_state():
    """Reset global state on initialization failure."""
    global _model, _config, _explainer, _initialization_complete
    _model = None
    _config = None
    _explainer = None
    _initialization_complete = False


# 애플리케이션 시작 시 설정 파일, 모델 로드 및 explainer 재생성 함수
async def load_resources(config_path: str = "app/models/model_config.yaml"):
    """
    Load configuration, AI model, and SHAP explainer on application startup.
    This function handles loading resources from a local file or Hugging Face Hub,
    and ensures global state is consistent even on failure.
    """
    global _model, _explainer, _config, _initialization_complete

    # 1. 설정 파일 로드
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
        print("설정 파일 로드 완료")
    except FileNotFoundError:
        print(f"오류: 설정 파일을 찾을 수 없습니다. 경로를 확인해주세요: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        print(f"설정 파일 로드 중 오류 발생: {e}")
        raise RuntimeError(f"Failed to load configuration file: {e}") from e

    # 2. Hugging Face Hub에서 모델 파일 다운로드 및 로드
    try:
        model_repo_id = _config["model"]["model_repo_id"]
        model_filename = _config["model"]["model_filename"]

        print(f"Hugging Face Hub에서 모델 파일 '{model_filename}' 다운로드 중...")
        model_path = hf_hub_download(
            repo_id=model_repo_id,
            filename=model_filename,
            repo_type="model"
        )
        print(f"모델 파일 다운로드 완료: {model_path}")

        _model = joblib.load(model_path)
        print("모델 로드 완료")

        _explainer = shap.TreeExplainer(_model)
        print("SHAP explainer 재생성 완료")

        _initialization_complete = True
        print("애플리케이션 초기화 완료")

    except KeyError as e:
        print(f"오류: 설정 파일에 필수 키가 누락되었습니다: {e}")
        _reset_global_state()
        raise RuntimeError(f"Configuration error: Missing key {e}") from e
    except Exception as e:
        print(f"모델 로드 또는 Explainer 재생성 중 오류 발생: {e}")
        _reset_global_state()
        raise RuntimeError(f"Failed to load model or regenerate explainer: {e}") from e


# 의존성 제공 함수 정의
def get_config():
    if _config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    return _config


def get_model():
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded during startup")
    return _model


def get_explainer():
    if _explainer is None:
        raise HTTPException(status_code=500, detail="Explainer not regenerated during startup")
    return _explainer


# 초기화 완료 상태를 반환하는 새로운 의존성 함수
def get_initialization_status():
    return _initialization_complete