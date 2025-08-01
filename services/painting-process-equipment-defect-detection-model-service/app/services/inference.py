import pandas as pd
from fastapi import HTTPException
from typing import Optional, Dict, Any

from app.services.utils import IssueLogInput

def _validate_input_ranges(input_data: IssueLogInput, config: dict) -> Optional[str]:
    """
    입력 데이터(전압, 전류, 온도)가 설정된 범위를 벗어나는지 검증합니다.

    Args:
        input_data (IssueLogInput): Pydantic 모델로 받은 입력 데이터.
        config (dict): 애플리케이션 설정 딕셔너리.

    Returns:
        Optional[str]: 유효성 검사 실패 시 해당하는 이슈 코드, 성공 시 None 반환.
    """
    try:
        voltage_range = config["validation"]["voltage_range"]
        current_range = config["validation"]["current_range"]
        temperature_range = config["validation"]["temperature_range"]
        log_issue_codes = config["logs"]["log_issue_codes"]
    except KeyError as e:
        print(f"오류: 설정 파일에 필수 키가 누락되었습니다: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: Missing key {e}") from e

    if not (voltage_range[0] <= input_data.voltage <= voltage_range[1]):
        return log_issue_codes["voltage_invalid"]
    elif not (current_range[0] <= input_data.current <= current_range[1]):
        return log_issue_codes["current_invalid"]
    elif not (temperature_range[0] <= input_data.temper <= temperature_range[1]):
        return log_issue_codes["temperature_invalid"]
    
    return None

def _predict_and_analyze(
    input_data: IssueLogInput,
    model,
    explainer,
    config: dict
) -> Optional[Dict]:
    """
    모델 예측 및 SHAP 분석을 수행하여 결함 원인을 파악합니다.

    Args:
        input_data (IssueLogInput): Pydantic 모델로 받은 입력 데이터.
        model: 로드된 AI 모델.
        explainer: 로드된 SHAP Explainer.
        config (dict): 애플리케이션 설정 딕셔너리.

    Returns:
        Optional[dict]: 문제 발생 시 분석 결과를 담은 딕셔너리, 문제 없을 시 None 반환.
    """
    try:
        input_columns = config["features"]["input_columns"]
        thickness_error_threshold = config["threshold"]["thickness_error_threshold"]
        shap_threshold = config["threshold"]["shap_threshold"]
        log_issue_codes = config["logs"]["log_issue_codes"]
    except KeyError as e:
        print(f"오류: 설정 파일에 필수 키가 누락되었습니다: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: Missing key {e}") from e

    # 입력 포맷 준비
    X_input_data = {
        input_columns[0]: input_data.voltage,
        input_columns[1]: input_data.current,
        input_columns[2]: input_data.temper
    }
    X_input = pd.DataFrame([X_input_data])

    # 예측
    if model is None:
        raise HTTPException(status_code=500, detail="Model not provided")
    pred_thick = model.predict(X_input)[0]

    # 예측 오차가 임계값 이하이면 로그 제외
    if abs(pred_thick - input_data.thick) / input_data.thick <= thickness_error_threshold:
        return None

    # SHAP 계산
    if explainer is None:
        raise HTTPException(status_code=500, detail="SHAP explainer not provided")

    shap_row = explainer(X_input)[0].values
    feature_names = list(X_input.columns)

    # SHAP 값 정리
    shap_dict = {f: abs(shap_row[feature_names.index(f)]) for f in feature_names}
    shap_raw_dict = {f: shap_row[feature_names.index(f)] for f in feature_names}
    shap_sum = sum(shap_dict.values())

    # 주요 원인 분석
    main_cause = max(shap_dict, key=lambda k: shap_dict[k])
    main_value = shap_raw_dict[main_cause]

    # 이슈 코드 생성
    if shap_sum < shap_threshold:
        issue = log_issue_codes["machine_issue"]
    else:
        label_map = {
            input_columns[0]: log_issue_codes["voltage_issue"],
            input_columns[1]: log_issue_codes["current_issue"],
            input_columns[2]: log_issue_codes["temperature_issue"]
        }
        base_issue_code = label_map[main_cause]
        issue = f"{base_issue_code}-{'HIGH' if main_value > 0 else 'LOW'}"

    # (옵션) SHAP 상세 값 출력용
    shap_detail = {
        f: round(shap_row[feature_names.index(f)], 5)
        for f in feature_names
    }

    return {
        "issue": issue,
        "shapDetail": shap_detail
    }

# MAIN ANALYSIS FUNCTION
def analyze_issue_log_api(
    input_data: IssueLogInput,
    model: Any,  # or specific model type if available
    explainer: Any,  # or shap.TreeExplainer
    config: dict
):
    """
    E-coating 공정 데이터를 분석하여 결함 및 원인을 파악하고 로그를 생성합니다.
    - 입력 데이터 유효성 검사, 모델 예측, SHAP 기반 원인 분석을 순차적으로 처리합니다.
    """
    # 설정이 로드되었는지 확인
    if config is None:
        print("오류: 설정이 인자로 전달되지 않았습니다.")
        raise HTTPException(status_code=500, detail="Configuration not provided")

    # 1. 입력 데이터 유효성 검사
    validation_issue = _validate_input_ranges(input_data, config)
    if validation_issue:
        issue = validation_issue
    else:
        # 2. 모델 예측 및 SHAP 분석
        analysis_result = _predict_and_analyze(input_data, model, explainer, config)
        if analysis_result is None:
            return None # 문제가 감지되지 않았으므로 None 반환

        issue = analysis_result["issue"]
        # shap_detail = analysis_result["shapDetail"]

    # 3. 최종 로그 데이터 포맷팅
    issue_with_time = f"{issue}-{input_data.timeStamp.isoformat()}"

    return {
        "machineId": input_data.machineId,
        "timeStamp": input_data.timeStamp.isoformat(),
        "thick": input_data.thick,
        "voltage": input_data.voltage,
        "current": input_data.current,
        "temper": input_data.temper,
        "issue": issue_with_time,
        "isSolved": False,
        # "shapDetail": shap_detail
    }