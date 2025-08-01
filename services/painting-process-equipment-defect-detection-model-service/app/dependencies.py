import joblib
import shap
import yaml
from fastapi import HTTPException
from typing import Optional, Dict, Any

# 전역 변수 초기화
_model: Optional[Any] = None
_config: Optional[Dict[str, Any]] = None
_explainer: Optional[Any] = None
_initialization_complete: bool = False  # 초기화 완료 상태를 추적하는 플래그

# 애플리케이션 시작 시 설정 파일, 모델 로드 및 explainer 재생성 함수
async def load_resources():
    global _model, _explainer, _config, _initialization_complete # 플래그를 global로 선언

    # 설정 파일 로드
    config_path = "app/models/model_config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
        print("설정 파일 로드 완료")
    except FileNotFoundError:
        print(f"오류: 설정 파일을 찾을 수 없습니다. 경로를 확인해주세요: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}") from None
    except Exception as e:
        print(f"설정 파일 로드 중 오류 발생: {e}")
        raise RuntimeError(f"Failed to load configuration file: {e}") from e

    # 모델 로드 (설정 파일에서 경로 읽기)
    try:
        model_path = _config["model"]["model_path"]
    except KeyError as e:
        print(f"오류: 설정 파일에 필수 키가 누락되었습니다: {e}")
        raise KeyError(f"Missing required key in configuration file: {e}") from e

    if not model_path:
        print("오류: 설정 파일에 모델 경로가 정의되지 않았습니다.")
        raise ValueError("Model path not defined in configuration file.")

    try:
        _model = joblib.load(model_path)
        print("모델 로드 완료")

        # 모델 로드 성공 후 Explainer 재생성
        _explainer = shap.TreeExplainer(_model)
        print("SHAP explainer 재생성 완료")

        _initialization_complete = True # 모든 리소스 로드 성공 시 플래그를 True로 설정
        print("애플리케이션 초기화 완료")

    except FileNotFoundError:
        print("오류: 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        raise FileNotFoundError("Model file not found at specified path.") from None
    except Exception as e:
        print(f"모델 로드 또는 explainer 재생성 중 오류 발생: {e}")
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