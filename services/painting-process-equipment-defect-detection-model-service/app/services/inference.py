import pandas as pd
from fastapi import HTTPException

from ..services.utils import IssueLogInput

def analyze_issue_log_api(
    input_data: IssueLogInput, # 입력 데이터 (Pydantic 모델 타입 힌트 유지)
    model,
    explainer,
    config: dict # 설정 딕셔너리
):
    # 이 함수 내부에서는 이제 인자로 받은 model, explainer, config를 사용합니다.

    # 설정이 로드되었는지 확인 (이제 인자로 받은 config 사용)
    if config is None:
        print("오류: 설정이 인자로 전달되지 않았습니다.")
        raise HTTPException(status_code=500, detail="Configuration not provided")


    # 입력값 unpack (IssueLogInput 객체에서 직접 접근)
    voltage = input_data.voltage
    current = input_data.current
    temper = input_data.temper
    thick = input_data.thick
    timestamp = input_data.timeStamp # 이미 datetime 객체

    # 설정에서 유효성 검사 범위 가져오기 등 (인자로 받은 config 사용)
    try:
        voltage_range = config["validation"]["voltage_range"]
        current_range = config["validation"]["current_range"]
        temperature_range = config["validation"]["temperature_range"]
        log_issue_codes = config["logs"]["log_issue_codes"]
        input_columns = config["features"]["input_columns"]
        thickness_error_threshold = config["threshold"]["thickness_error_threshold"]
        shap_threshold = config["threshold"]["shap_threshold"]
    except KeyError as e:
        print(f"오류: 설정 파일에 필수 키가 누락되었습니다: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: Missing key {e}")

    # 유효성 검사
    if not (voltage_range[0] < voltage < voltage_range[1]):
        issue = log_issue_codes["voltage_invalid"]
    elif not (current_range[0] < current < current_range[1]):
        issue = log_issue_codes["current_invalid"]
    elif not (temperature_range[0] < temper < temperature_range[1]):
        issue = log_issue_codes["temperature_invalid"]
    else:
        # 입력 포맷 준비
        X_input_data = {
            input_columns[0]: voltage,
            input_columns[1]: current,
            input_columns[2]: temper
        }
        X_input = pd.DataFrame([X_input_data])

        # 예측 (인자로 받은 model 사용)
        if model is None:
            print("오류: 모델이 인자로 전달되지 않았습니다.")
            raise HTTPException(status_code=500, detail="Model not provided")

        pred_thick = model.predict(X_input)[0]

        # 예측 오차가 임계값 이하이면 로그 제외
        if abs(pred_thick - thick) / thick <= thickness_error_threshold:
            return None

        # SHAP 계산 (인자로 받은 explainer 사용)
        if explainer is None:
            print("오류: SHAP explainer가 인자로 전달되지 않았습니다.")
            raise HTTPException(status_code=500, detail="SHAP explainer not provided")

        shap_row = explainer(X_input)[0].values
        feature_names = list(X_input.columns)

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
            label_map = {
                input_columns[0]: log_issue_codes["voltage_issue"],
                input_columns[1]: log_issue_codes["current_issue"],
                input_columns[2]: log_issue_codes["temperature_issue"]
            }
            base_issue_code = label_map[main_cause]
            issue = f"{base_issue_code}-{'HIGH' if main_value > 0 else 'LOW'}"

        # # (옵션) SHAP 상세 값 출력용
        # shap_detail = {
        #     f: round(shap_row[feature_names.index(f)], 5)
        #     for f in feature_names
        # }

    issue_with_time = f"{issue}-{timestamp.isoformat()}"

    return {
        "machineId": input_data.machineId,
        "timeStamp": timestamp.isoformat(),
        "thick": thick,
        "voltage": voltage,
        "current": current,
        "temper": temper,
        "issue": issue_with_time,
        "isSolved": False,
        # "shapDetail": shap_detail
    }