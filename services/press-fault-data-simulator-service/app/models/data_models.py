from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PredictionRequest(BaseModel):
    """
    고장 예측 API(/predict)로 전송할 요청 본문의 데이터 모델.
    이 시뮬레이터가 생성해야 할 데이터의 최종 형태.
    """

    AI0_Vibration: List[float] = Field(..., description="시계열 진동 데이터 리스트")
    AI1_Vibration: List[float] = Field(..., description="진동 데이터 리스트")
    AI2_Current: List[float] = Field(..., description="전류 데이터 리스트")

    @classmethod
    def from_csv_data(cls, df_data) -> "PredictionRequest":
        """
        pandas DataFrame에서 PredictionRequest 객체 생성

        Args:
            df_data: pandas DataFrame with columns ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']

        Returns:
            PredictionRequest: API 요청용 데이터 객체
        """
        required_columns = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]

        if df_data is None or df_data.empty:
            raise ValueError("DataFrame이 비어있습니다.")

        missing_columns = [
            col for col in required_columns if col not in df_data.columns
        ]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")

        data_dict = {
            "AI0_Vibration": df_data["AI0_Vibration"].tolist(),
            "AI1_Vibration": df_data["AI1_Vibration"].tolist(),
            "AI2_Current": df_data["AI2_Current"].tolist(),
        }
        return cls(**data_dict)


class PredictionResult(BaseModel):
    """
    고장 예측 API(/predict)로부터 수신할 응답 결과의 데이터 모델.
    이 시뮬레이터가 수신하고 처리해야 할 데이터의 형태.
    """

    prediction: str
    reconstruction_error: float
    is_fault: bool
    fault_probability: Optional[float] = None
    attribute_errors: Optional[Dict[str, float]] = None
