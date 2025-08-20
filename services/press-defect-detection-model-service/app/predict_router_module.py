from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Dict
import tempfile
import os
import logging
from datetime import datetime

from services.inference import InferenceService  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])

inference_service: Optional[InferenceService] = None

def set_inference_service(service: InferenceService):
    global inference_service
    inference_service = service

class SingleImageRequest(BaseModel):
    image_base64: str
    image_name: Optional[str] = None

    @validator('image_base64')
    def validate_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError('유효하지 않은 base64 이미지입니다.')
        return v

class MultiImageRequest(BaseModel):
    images: List[Dict[str, str]]

    @validator('images')
    def validate_images(cls, v):
        if not v or len(v) == 0:
            raise ValueError('최소 1개의 이미지가 필요합니다.')
        if len(v) > 50:
            raise ValueError('최대 50장까지만 처리 가능합니다.')
        return v

class InspectionBatchRequest(BaseModel):
    inspection_id: str
    images: List[Dict[str, str]]

    @validator('inspection_id')
    def validate_inspection_id(cls, v):
        if not v or not v.strip():
            raise ValueError('inspection_id는 필수입니다.')
        return v.strip()

    @validator('images')
    def validate_inspection_images(cls, v):
        if len(v) != 21:
            logger.warning(f"권장 이미지 수는 21장이지만 {len(v)}장이 제공되었습니다.")
        return v

@router.post("/", summary="단일 이미지 구멍 탐지")
async def predict_single_image(request: SingleImageRequest):
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    try:
        result = inference_service.predict_single_image(
            image_input=request.image_base64,
            input_type="base64"
        )
        if result['success']:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))
    except Exception as e:
        logger.error(f"단일 이미지 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}") from e

@router.post("/multi", summary="다중 이미지 품질 검사")
async def predict_multi_images(request: MultiImageRequest):
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    try:
        image_inputs = []
        for idx, img_data in enumerate(request.images):
            image_inputs.append({
                "image": img_data.get("image", ""),
                "type": "base64",
                "name": img_data.get("name", f"image_{idx}")
            })
        result = inference_service.predict_multiple_images(image_inputs)
        if result['success']:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))
    except Exception as e:
        logger.error(f"다중 이미지 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}") from e

@router.post("/inspection", summary="검사 배치 품질 검사")
async def predict_inspection_batch(request: InspectionBatchRequest):
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    try:
        image_inputs = []
        for idx, img_data in enumerate(request.images):
            image_inputs.append({
                "image": img_data.get("image", ""),
                "type": "base64",
                "name": img_data.get("name", f"cam_{idx}")
            })
        result = inference_service.predict_inspection_batch(
            inspection_id=request.inspection_id,
            images=image_inputs
        )
        if result['success']:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))
    except Exception as e:
        logger.error(f"검사 배치 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}") from e

_file_param = File(...)

@router.post("/file", summary="파일 업로드 예측")
async def predict_file_upload(file: UploadFile = _file_param):
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ''
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}"
        )

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)

        result = inference_service.predict_single_image(
            image_input=temp_file.name,
            input_type="file_path"
        )
        if result['success']:
            result['file_info'] = {
                'filename': file.filename,
                'file_size': len(content),
                'content_type': file.content_type
            }
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))
    except Exception as e:
        logger.error(f"파일 업로드 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}") from e
    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logging.getLogger(__name__).warning(f"임시 파일 삭제 실패: {e}")

@router.get("/service-info", summary="추론 서비스 정보")
async def get_service_info():
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    return inference_service.get_service_info()

@router.get("/health", summary="예측 서비스 헬스체크")
async def prediction_health():
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    return {
        "status": "healthy",
        "service": "prediction",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": inference_service.yolo_model.model_loaded if inference_service.yolo_model else False
    }