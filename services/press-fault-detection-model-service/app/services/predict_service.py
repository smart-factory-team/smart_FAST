import numpy as np
import logging

from app.core.model_cache import model_cache
from app.schemas.input import SensorData, SEQUENCE_LENGTH

logger = logging.getLogger(__name__)

def predict(data: SensorData) -> dict:
    """_summary_
        입력된 센서 데이터를 기반으로 고장 예측
    Args:
        data (SensorData): 상/하부 진동과 전류값의 List

    Returns:
        dict: 예측한 정상/고장 값, 재구성 오차 값, is_fault boolean 값
    """
    try:
        #1. 모델 로드
        model = model_cache.get("model")
        scaler = model_cache.get("scaler")
        threshold = model_cache.get("threshold")
        
        if not all([model, scaler, threshold is not None]):
            raise RuntimeError("모델, 스케일러 또는 임계값이 로드되지 않았습니다. 서버 로그를 확인하세요.")
        
        #2. 데이터 전처리 : 절댓값
        AI0_abs = [abs(x) for x in data.AI0_Vibration]
        AI1_abs = [abs(x) for x in data.AI1_Vibration]
        AI2_abs = [abs(x) for x in data.AI2_Current]
        
        # 3. 입력 데이터 변환 : 3개의 리스트 -> (N, 3) 형태의 numpy 배열
        input_array = np.vstack([
            AI0_abs,
            AI1_abs,
            AI2_abs
        ]).T
        
        # 4. 데이터 스케일링
        scaler_data = scaler.transform(input_array)
        
        # 5. 예측에 사용할 시퀀스 추출
        last_sequence = scaler_data[-SEQUENCE_LENGTH:]
        # 모델 입력 형태로 차원 확장: (20, 3) -> (1, 20, 3)
        sequence_for_prediction = np.expand_dims(last_sequence, axis=0)
        
        # 6. 예측, 복원 오차 계산
        reconstructed_sequence = model.predict(sequence_for_prediction)
        error = np.mean(np.power(flatten(reconstructed_sequence) - flatten(sequence_for_prediction), 2), axis=1)
        
        # 7. 고장 여부 판정
        is_fault = error>threshold
        prediction_result = "고장" if is_fault else "정상"
        logger.info(threshold)
        return {
            "prediction": prediction_result,
            "reconstruction_error": float(error),
            "is_fault": is_fault
        }
    except Exception as e:
        logger.error(f'예측 서비스 실행 중 예외 발생: {e}', exc_info=True)
        raise e
    

# 모델 출력값과 입력값의 재구성 오차를 계산하기위해 데이터 차원을 감소시키는 함수
def flatten(X):
    flattened = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened[i] = X[i, X.shape[1]-1, :]
    return(flattened)