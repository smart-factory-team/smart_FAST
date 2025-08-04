from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Response
from datetime import datetime
from typing import Dict, Any, List
import csv
import codecs

from app.services.inference import analyze_issue_log_api as run_analysis
from app.services.utils import save_issue_log, IssueLogInput
from app.dependencies import get_config, get_model, get_explainer

router = APIRouter()

@router.post("/")
async def predict_issue(
    input_data: IssueLogInput,
    config: Dict[str, Any] = Depends(get_config),
    model: Any = Depends(get_model),
    explainer: Any = Depends(get_explainer)
):
    """
    E-coating 공정 데이터를 분석하여 결함 및 원인을 파악하고 로그를 생성합니다.
    분석 결과 로그는 파일에 저장됩니다.
    예측 오차가 임계값 이내인 경우 204 No Content를 반환합니다.
    """
    try:
        # run_analysis 함수를 호출하여 분석을 수행합니다.
        result = run_analysis(input_data, model, explainer, config)
    except HTTPException as e:
        # 의존성에서 발생한 HTTPException은 그대로 전달합니다.
        raise e
    except Exception as e:
        # 예상치 못한 다른 오류가 발생하면 500 오류를 반환합니다.
        print(f"분석 중 오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"예상치 못한 오류가 발생했습니다: {str(e)}"
        )

    # analyze_issue_log_api가 None을 반환하면(이슈가 없으면) 204 No Content를 반환합니다.
    if result is None:
        return Response(status_code=204)
    
    # 이슈가 감지되면 로그를 저장하고 결과를 반환합니다.
    try:
        save_issue_log(result, config)
    except Exception as e:
        print(f"이슈 로그를 저장하는 중에 오류가 발생했습니다: {e}")
        # 로그 저장 실패는 Critical 오류가 아니므로 200 OK를 반환하되, 경고를 포함합니다.
        return {"predictions": [result], "warning": f"로그 파일 저장에 실패했습니다: {str(e)}"}
    
    return {"predictions": [result]}

@router.post("/file")
async def predict_issue_from_file(
    file: UploadFile = File(...),
    config: Dict[str, Any] = Depends(get_config),
    model: Any = Depends(get_model),
    explainer: Any = Depends(get_explainer)
):
    """
    CSV 파일을 업로드하여 E-coating 공정 데이터의 결함 및 원인을 파악하고 로그를 생성합니다.
    """
    # CSV 파일만 허용합니다.
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="잘못된 파일 형식입니다. CSV 파일만 허용됩니다."
        )

    try:
        # 설정 파일에서 컬럼 매핑 정보를 가져옵니다.
        try:
            column_mapping = config["file_upload"]["column_mapping"]
        except KeyError:
            raise HTTPException(
                status_code=500,
                detail="설정 파일에서 컬럼 매핑 정보를 찾을 수 없습니다."
            )

        results: List[Dict[str, Any]] = []
        
        # 파일을 스트리밍으로 처리합니다.
        # UTF-8-sig는 BOM(Byte Order Mark)을 처리할 수 있습니다.
        csv_reader = csv.reader(codecs.iterdecode(file.file, 'utf-8-sig'))
        
        # 헤더(컬럼명)를 읽어옵니다.
        header = next(csv_reader)
        
        # 매핑된 컬럼명을 기준으로 필수 컬럼이 모두 있는지 확인합니다.
        mapped_header = [column_mapping.get(h, h) for h in header]
        required_columns = list(column_mapping.values())
        if not all(col in mapped_header for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV 파일에 필수 컬럼이 누락되었습니다: {required_columns}"
            )

        print(f"{file.filename} 파일 처리를 시작합니다...")

        for index, row in enumerate(csv_reader):
            # 각 행을 {컬럼명: 값} 형태의 딕셔너리로 변환합니다.
            row_data = {mapped_header[i]: val for i, val in enumerate(row)}
            
            try:
                # timeStamp가 datetime 객체로 변환되도록 처리
                if 'timeStamp' in row_data and isinstance(row_data['timeStamp'], str):
                    row_data['timeStamp'] = datetime.fromisoformat(row_data['timeStamp'])
                
                # 데이터 타입을 Pydantic 모델에 맞게 변환합니다.
                # 예를 들어, thick, voltage 등은 float으로 변환해야 합니다.
                for key, value in row_data.items():
                    if key in ['thick', 'voltage', 'current', 'temper']:
                        row_data[key] = float(value)

                input_data = IssueLogInput(**row_data)
                result = run_analysis(input_data, model, explainer, config)
                
                if result:  # 예측 오차가 임계값 이내가 아니면
                    save_issue_log(result, config)
                    results.append(result)

            except HTTPException as e:
                results.append({"row_index": index, "status": "failed", "error": e.detail})
                print(f"{index}행 처리 중 HTTPException 발생: {e.detail}")
            except Exception as e:
                results.append({"row_index": index, "status": "failed", "error": str(e)})
                print(f"{index}행 처리 중 오류 발생: {e}")

        if not results:
            return {"message": "감지된 이슈가 없거나 모든 행 처리 중 오류가 발생했습니다."}

        return {"filename": file.filename, "predictions": results}
    except Exception as e:
        print(f"파일 처리 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"파일 처리 실패: {str(e)}"
        )
    finally:
        # 파일 핸들을 닫아줍니다.
        await file.close()