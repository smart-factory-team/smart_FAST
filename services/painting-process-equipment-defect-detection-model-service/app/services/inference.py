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
    else:
        return None

def _predict_and_analyze(
    input_data: IssueLogInput,
    model: Any,
    explainer: Any,
    config: dict
) -> Optional[Dict[str, Any]]:
    """
    모델 예측 및 SHAP 기반 원인 분석을 수행하고, 결과를 사전 형태로 반환합니다.

    Args:
        input_data (IssueLogInput): Pydantic 모델로 받은 입력 데이터.
        model (Any): 로드된 AI 모델 객체.
        explainer (Any): SHAP Explainer 객체.
        config (dict): 애플리케이션 설정 딕셔너리.

    Returns:
        Optional[Dict[str, Any]]: 분석 결과 사전. 문제가 감지되지 않으면 None 반환.
    """
    try:
        # 설정값 가져오기
        input_columns = config["features"]["input_columns"]
        thickness_error_threshold = config["threshold"]["thickness_error_threshold"]
        shap_threshold = config["threshold"]["shap_threshold"]
        log_issue_codes = config["logs"]["log_issue_codes"]
    except KeyError as e:
        print(f"오류: 설정 파일에 필수 키가 누락되었습니다: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: Missing key {e}") from e

    # Pandas DataFrame으로 변환
    df_input = pd.DataFrame([input_data.model_dump()], columns=input_columns)

    # 모델 예측
    y_pred = model.predict(df_input)[0]

    # 예측 오차 계산
    prediction_error = abs(input_data.thick - y_pred)

    # 예측 오차가 임계값 이내면 문제 없음으로 판단
    if prediction_error < thickness_error_threshold:
        return None

    # SHAP 분석
    shap_values = explainer.shap_values(df_input)
    shap_row = shap_values[0] # 첫 번째 샘플에 대한 SHAP 값
    feature_names = input_columns

    # SHAP 값 정리
    shap_dict = {f: abs(shap_row[feature_names.index(f)]) for f in feature_names}
    shap_raw_dict = {f: shap_row[feature_names.index(f)] for f in feature_names}
    shap_sum = sum(shap_dict.values())

    # 주요 원인 분석
    main_cause = max(shap_dict, key=shap_dict.get)
    main_value = shap_raw_dict[main_cause]

    # 이슈 코드 생성 (설정 값 사용)
    machine_issue_code = log_issue_codes["machine_issue"]

    if shap_sum < shap_threshold:
        issue = machine_issue_code
    else:
        issue_map = config["logs"]["issue_map"]
        base_issue_code = issue_map.get(main_cause, log_issue_codes["machine_issue"])
        issue = f"{base_issue_code}-{'HIGH' if main_value > 0 else 'LOW'}"
    
    # SHAP 상세 값 (옵션)
    shap_detail = {f: round(shap_row[feature_names.index(f)], 5) for f in feature_names}

    return {
        "predictionError": prediction_error,
        "issue": issue,
        "shapDetail": shap_detail
    }

# MAIN ANALYSIS FUNCTION
def analyze_issue_log_api(
    input_data: IssueLogInput,
    model: Any,
    explainer: Any,
    config: dict
):
    """
    E-coating 공정 데이터를 분석하여 결함 및 원인을 파악하고 로그를 생성합니다.
    - 입력 데이터 유효성 검사, 모델 예측, SHAP 기반 원인 분석을 순차적으로 처리합니다.
    """
    if config is None:
        print("오류: 설정이 인자로 전달되지 않았습니다.")
        raise HTTPException(status_code=500, detail="Configuration not provided")

    # 입력 데이터 유효성 검사
    validation_issue_code = _validate_input_ranges(input_data, config)
    if validation_issue_code:
        # 유효성 검사 실패 시, 평면적인 로그 데이터 포맷으로 반환
        log_data = input_data.model_dump()
        log_data["issue"] = validation_issue_code
        log_data["isSolved"] = False # 기본값 False
        log_data["timeStamp"] = log_data["timeStamp"].isoformat() # 직렬화 문제 해결
        return log_data
    
    # 모델 예측 및 SHAP 분석
    analysis_result = _predict_and_analyze(input_data, model, explainer, config)
    if analysis_result is None:
        return None # 문제가 감지되지 않았으므로 None 반환

    # 최종 로그 데이터 포맷팅
    log_data = input_data.model_dump()
    log_data["issue"] = f"{analysis_result['issue']}"
    log_data["isSolved"] = False # 기본값 False
    log_data["timeStamp"] = log_data["timeStamp"].isoformat() # 직렬화 문제 해결

    return log_data