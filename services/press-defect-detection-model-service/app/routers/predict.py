from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import tempfile
import os
import logging
from datetime import datetime
from starlette.concurrency import run_in_threadpool

from services.inference import InferenceService

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(prefix="/predict", tags=["prediction"])

# 전역 변수 (main.py에서 설정)
inference_service: Optional[InferenceService] = None

def set_inference_service(service: InferenceService):
    """메인에서 추론 서비스 설정"""
    global inference_service
    inference_service = service

# Pydantic 모델 정의
class SingleImageRequest(BaseModel):
    """단일 이미지 예측 요청"""
    image_base64: str
    image_name: Optional[str] = None
    
    @validator('image_base64')
    def validate_base64(cls, v):
        if not v or len(v) < 100:  # 너무 짧은 base64는 무효
            raise ValueError('유효하지 않은 base64 이미지입니다.')
        return v

class MultiImageRequest(BaseModel):
    """다중 이미지 예측 요청"""
    images: List[Dict[str, str]]  # [{"image": "base64", "name": "optional"}]
    
    @validator('images')
    def validate_images(cls, v):
        if not v or len(v) == 0:
            raise ValueError('최소 1개의 이미지가 필요합니다.')
        if len(v) > 50:  # 최대 50장 제한
            raise ValueError('최대 50장까지만 처리 가능합니다.')
        return v

class InspectionBatchRequest(BaseModel):
    """검사 배치 예측 요청"""
    inspection_id: str
    images: List[Dict[str, str]]  # 21장의 이미지
    
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

# API 엔드포인트들

@router.post("/", summary="단일 이미지 구멍 탐지")
async def predict_single_image(request: SingleImageRequest):
    """
    단일 이미지에서 구멍 탐지
    
    - **image_base64**: Base64 인코딩된 이미지
    - **image_name**: 선택적 이미지 이름
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    
    try:
        result = await run_in_threadpool(
            inference_service.predict_single_image,
            request.image_base64,
            "base64",
        )
        
        if result['success']:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))
            
    except Exception as e:
        logger.error(f"단일 이미지 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

@router.post("/multi", summary="다중 이미지 품질 검사")
async def predict_multi_images(request: MultiImageRequest):
    """
    다중 이미지로 품질 검사 수행
    
    - **images**: 이미지 리스트 [{"image": "base64", "name": "optional"}]
    - 적응형 임계값을 적용하여 품질 검사 수행
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    
    try:
        # 이미지 데이터 준비
        image_inputs = []
        for idx, img_data in enumerate(request.images):
            image_inputs.append({
                "image": img_data.get("image", ""),
                "type": "base64",
                "name": img_data.get("name", f"image_{idx}")
            })
        
        result = await run_in_threadpool(
            inference_service.predict_multiple_images,
            image_inputs,
        )
        
        if result['success']:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))
            
    except Exception as e:
        logger.error(f"다중 이미지 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

@router.post("/inspection", summary="검사 배치 품질 검사")
async def predict_inspection_batch(request: InspectionBatchRequest):
    """
    검사 배치 품질 검사 (권장: 21장 이미지)
    
    - **inspection_id**: 검사 ID
    - **images**: 이미지 리스트 (권장 21장)
    - 최종 Pass/Reject 판정 수행
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    
    try:
        # 이미지 데이터 준비
        image_inputs = []
        for idx, img_data in enumerate(request.images):
            image_inputs.append({
                "image": img_data.get("image", ""),
                "type": "base64", 
                "name": img_data.get("name", f"cam_{idx}")
            })
        
        result = await run_in_threadpool(
            inference_service.predict_inspection_batch,
            request.inspection_id,
            image_inputs,
        )
        
        if result['success']:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))
            
    except Exception as e:
        logger.error(f"검사 배치 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

@router.post("/file", summary="파일 업로드 예측")
async def predict_file_upload(file: UploadFile = File(...)):
    """
    파일 업로드를 통한 단일 이미지 예측
    
    - **file**: 업로드할 이미지 파일 (JPEG, PNG 지원)
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    
    # 파일 확장자 검증
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ''
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}"
        )
    
    # 임시 파일로 저장
    temp_file = None
    try:
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        
        # 파일 내용 저장
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        result = await run_in_threadpool(
            inference_service.predict_single_image,
            temp_file.name,
            "file_path",
        )

        if result['success']:
            # 파일 정보 추가
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
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")
    
    finally:
        # 임시 파일 정리
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")

@router.get("/service-info", summary="추론 서비스 정보")
async def get_service_info():
    """추론 서비스 정보 조회"""
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    
    return inference_service.get_service_info()

# 헬스체크 및 상태 확인
@router.get("/health", summary="예측 서비스 헬스체크")
async def prediction_health():
    """예측 서비스 상태 확인"""
    if not inference_service:
        raise HTTPException(status_code=503, detail="추론 서비스가 준비되지 않았습니다.")
    
    return {
        "status": "healthy",
        "service": "prediction",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": inference_service.yolo_model.model_loaded if inference_service.yolo_model else False
    }