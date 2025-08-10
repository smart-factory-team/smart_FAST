import numpy as np
import logging

from app.core.model_cache import model_cache
from app.schemas.input import SensorData, SEQUENCE_LENGTH
from numpy.lib.stride_tricks import as_strided

logger = logging.getLogger(__name__)
FEATURE_NAMES = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]


def predict_press_fault(data: SensorData) -> dict:
    """_summary_
        입력된 센서 데이터를 기반으로 고장 예측
    Args:
        data (SensorData): 상/하부 진동과 전류 센서 데이터

    Returns:
        dict: 예측한 정상/고장 값, 재구성 오차 값, is_fault boolean 값
    """
    try:
        # 1. 모델 로드
        model = model_cache.get("model")
        scaler = model_cache.get("scaler")
        threshold = model_cache.get("threshold")

        if not all([model, scaler, threshold is not None]):
            raise RuntimeError(
                "모델, 스케일러 또는 임계값이 로드되지 않았습니다. 서버 로그를 확인하세요."
            )

        # 2. 데이터 전처리
        AI0_abs = [abs(x) for x in data.AI0_Vibration]
        AI1_abs = [abs(x) for x in data.AI1_Vibration]
        AI2_abs = [abs(x) for x in data.AI2_Current]

        # 3개의 리스트 -> (N, 3) 형태의 numpy 배열
        input_array = np.vstack([AI0_abs, AI1_abs, AI2_abs]).T

        # 3. 데이터 스케일링
        scaler_data = scaler.transform(input_array)

        # 4. 시퀀스 데이터 생성
        sequences = create_sequences(scaler_data, SEQUENCE_LENGTH)
        if len(sequences) == 0:
            raise ValueError(
                f"입력 데이터 길이({len(scaler_data)})가 시퀀스 길이 ({SEQUENCE_LENGTH})보다 짧아 예측을 수행할 수 없습니다."
            )

        # 5. 예측, 복원 오차 계산
        reconstructed_sequence = model.predict(sequences)

        # 6-1. 속성별 제곱 오차 계산 (아직 평균 내지 않음)
        # 이 시점에서 per_feature_squared_errors는 (N, 3) 형태의 배열
        # 각 행은 시퀀스, 각 열은 속성(피처)의 제곱 오차
        original_last_steps = flatten(sequences)
        reconstructed_last_steps = flatten(reconstructed_sequence)
        per_feature_squared_errors = np.power(
            reconstructed_last_steps - original_last_steps, 2
        )

        # 6-2. 기존 방식대로 시퀀스별 최종 오차 계산 (속성들을 평균)
        errors = np.mean(per_feature_squared_errors, axis=1)

        # 7. 최종 고장 여부 판정
        max_error = np.max(errors)
        is_fault = max_error > threshold
        prediction_result = "고장" if is_fault else "정상"

        # 8. 고장일 경우, 원인 분석
        attribute_errors_dict = None
        if is_fault:
            # 가장 큰 오차를 발생시킨 시퀀스의 인덱스 찾기
            worst_sequence_index = np.argmax(errors)

            # 해당 시퀀스의 속성별 오차 값
            faulty_attribute_errors = per_feature_squared_errors[worst_sequence_index]

            # 속성 이름과 오차 값을 딕셔너리로 매핑
            attribute_errors_dict = dict(
                zip(FEATURE_NAMES, faulty_attribute_errors.astype(float))
            )

        return {
            "prediction": prediction_result,
            "reconstruction_error": float(max_error),
            "is_fault": is_fault,
            "attribute_errors": attribute_errors_dict,
        }
    except Exception as e:
        logger.error(f"예측 서비스 실행 중 예외 발생: {e}", exc_info=True)
        raise e


# 모델 출력값과 입력값의 재구성 오차를 계산하기위해 데이터 차원을 감소시키는 함수
def flatten(X):
    if X.shape[1] < 1:
        return np.empty((X.shape[0], X.shape[2]))
    flattened = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened[i] = X[i, X.shape[1] - 1, :]
    return flattened


def create_sequences(data, seq_length):
    """2D배열 데이터로부터 슬라이딩 윈도우 방식으로 시퀀스를 생성"""
    if len(data) < seq_length:
        return np.array([])
    sequences = len(data) - seq_length + 1
    return as_strided(
        data,
        shape=(sequences, seq_length, data.shape[1]),
        strides=(data.strides[0], data.strides[0], data.strides[1]),
    )
