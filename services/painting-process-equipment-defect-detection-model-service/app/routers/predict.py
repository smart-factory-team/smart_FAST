from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import Response
import pandas as pd
from io import StringIO
from datetime import datetime

from app.services.inference import analyze_issue_log_api as run_analysis
from app.services.utils import save_issue_log, IssueLogInput
from app.dependencies import get_config, get_model, get_explainer

router = APIRouter()

@router.post("/")
async def predict_issue(
    input_data: IssueLogInput,
    config: dict = Depends(get_config),
    model = Depends(get_model),
    explainer = Depends(get_explainer)
):
    """
    E-coating 공정 데이터를 분석하여 결함 및 원인을 파악하고 로그를 생성합니다.
    분석 결과 로그는 파일에 저장됩니다.
    예측 오차가 임계값 이내인 경우 204 No Content를 반환합니다.
    """
    result = run_analysis(input_data, model, explainer, config)

    # analyze_issue_log_api 함수가 None을 반환하면 (오차 10% 이내), 로그를 생성하지 않음
    if result is None:
        # return {"message": "Prediction error within threshold, no issue logged."}
        return Response(status_code=204) # 204 No Content
    
    # 분석 결과가 있으면 로그를 파일에 저장
    save_issue_log(result, config)

    return result

@router.post("/file")
async def predict_issue_from_file(
    file: UploadFile = File(...),
    config: dict = Depends(get_config),
    model = Depends(get_model),
    explainer = Depends(get_explainer)
):
    """
    CSV 파일 업로드를 통해 E-coating 공정 데이터를 분석하고 예측합니다.
    파일 내 각 행에 대해 예측을 수행하며, 문제가 발견되면 로그를 생성합니다.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed for file prediction.")
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty.")

        results = []
        for index, row in df.iterrows():
            # CSV 행 데이터를 IssueLogInput Pydantic 모델로 변환
            try:
                # timeStamp가 datetime 객체로 변환되도록 처리
                if 'timeStamp' in row and not isinstance(row['timeStamp'], datetime):
                    row['timeStamp'] = pd.to_datetime(row['timeStamp'])
                input_data = IssueLogInput(**row.to_dict())
                result = run_analysis(input_data, model, explainer, config)
                if result: # 예측 오차가 임계값 이내가 아니면
                    save_issue_log(result, config)
                    results.append(result)
            except Exception as e:
                # 개별 행 처리 중 에러 발생 시 해당 에러를 기록하고 다음 행으로 진행
                results.append({"row_index": index, "status": "failed", "error": str(e)})
                print(f"Error processing row {index}: {e}")

        if not results:
            return {"message": "No issues detected or all rows processed with errors."}

        return {"filename": file.filename, "predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {e}") from e