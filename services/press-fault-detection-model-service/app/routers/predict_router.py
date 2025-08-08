from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import io
import logging

from app.schemas.input import SensorData
from app.schemas.output import PredictionResponse
from app.services.predict_service import predict

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["prediction"]
)

@router.post("", response_model=PredictionResponse)
async def predict_fault(request_body: SensorData):
    
    try:
        result = predict(data=request_body)
        return PredictionResponse(**result)
    
    except RuntimeError as e:
        # 모델이 로드되지 않았을 경우
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        # 그 외 예측 과정에서 발생할 수 있는 모든 오류
        raise HTTPException(status_code=500, detail=f"예측 처리 중 오류 발생: {str(e)}") from e 


@router.post("/file", response_model=PredictionResponse)
async def predict_fault_from_file(file: UploadFile = File(...)):
    logger.info("start predict file")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다.")
    
    try:
        # CSV 파일 읽기
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df = df.dropna(axis=0)  # NaN 포함된 행 삭제
        # CSV 데이터를 SensorData 형식으로 변환
        required_columns = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV 파일에 필요한 컬럼이 없습니다: {missing_columns}"
            )
        
        logger.info("CSV to DF")
        
        # DataFrame을 SensorData로 변환
        sensor_data = SensorData(
            AI0_Vibration=df['AI0_Vibration'].tolist(),
            AI1_Vibration=df['AI1_Vibration'].tolist(),
            AI2_Current=df['AI2_Current'].tolist()
        )
        
        logger.info("DataFrame to SensorData")
        
        # 기존 predict 함수 사용
        result = predict(data=sensor_data)
        logger.info("predict")
        return PredictionResponse(**result)
        
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail="파일을 읽을 수 없습니다. UTF-8 인코딩을 확인해주세요.") from e
    except pd.errors.EmptyDataError as e:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어있습니다.") from e
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV 파일 형식이 올바르지 않습니다: {str(e)}") from e
    except ValueError as e:
        # SensorData 유효성 검사 실패
        raise HTTPException(status_code=400, detail=f"데이터 유효성 검사 실패: {str(e)}") from e
    except RuntimeError as e:
        # 모델이 로드되지 않았을 경우
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        # 그 외 예측 과정에서 발생할 수 있는 모든 오류
        raise HTTPException(status_code=500, detail=f"파일 예측 처리 중 오류 발생: {str(e)}") from e
    
