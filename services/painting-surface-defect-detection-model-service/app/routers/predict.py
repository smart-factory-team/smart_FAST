from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import base64
import logging

from app.services.inference import PaintingSurfaceDefectDetectionService

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# 전역 서비스 인스턴스 (main.py에서 설정됨)
detection_service: Optional[PaintingSurfaceDefectDetectionService] = None


from pydantic import BaseModel, validator, Field

class PredictionRequest(BaseModel):
    """예측 요청 스키마 (Base64 방식)"""
    image_base64: str = Field(..., min_length=100, description="Base64 인코딩된 이미지 데이터")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="신뢰도 임계값 (0.0-1.0)")
    
    @validator('image_base64')
    def validate_base64(cls, v):
        """Base64 데이터 유효성 검사"""
        if not v or len(v) < 100:
            raise ValueError("Base64 이미지 데이터가 유효하지 않습니다")
        try:
            import base64
            base64.b64decode(v)
        except Exception:
            raise ValueError("유효하지 않은 Base64 인코딩입니다")
        return v


class PredictionResponse(BaseModel):
    """예측 응답 스키마"""
    predictions: List[Dict[str, Any]]
    image_shape: List[int]
    confidence_threshold: float
    timestamp: str
    model_source: str


def get_detection_service() -> PaintingSurfaceDefectDetectionService:
    """의존성 주입을 위한 서비스 가져오기"""
    if detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not available")
    return detection_service


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="이미지 파일"),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0, description="신뢰도 임계값 (0.0-1.0)")
):
    """
    도장 표면 결함 탐지 (파일 업로드 방식)
    
    Args:
        image: 업로드된 이미지 파일
        confidence_threshold: 신뢰도 임계값
    
    Returns:
        결함 탐지 결과
    """
    try:
        # 파일 유효성 검사
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
        
        # 파일 크기 제한 (10MB)
        if image.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다")
        
        # 이미지 데이터 읽기
        image_data = await image.read()
        
        # 예측 수행
        result = predict_anomaly(image_data, confidence_threshold)
        
        logger.info(f"Prediction completed successfully for {image.filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/base64", response_model=PredictionResponse)
def predict_base64(data: PredictionRequest):
    """
    도장 표면 결함 탐지 (Base64 방식)
    
    Args:
        data: Base64 인코딩된 이미지 데이터와 설정 정보
    
    Returns:
        결함 탐지 결과
    """
    try:
        # Base64 디코딩 (Pydantic에서 이미 검증됨)
        image_data = base64.b64decode(data.image_base64)
        
        # 예측 수행
        result = predict_anomaly(image_data, data.confidence_threshold)
        
        logger.info("Base64 prediction completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in base64 prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Base64 prediction failed: {str(e)}")


def predict_anomaly(image_data: bytes, confidence_threshold: Optional[float] = None):
    """도장 표면 결함 탐지 (동기 함수)"""
    if detection_service is None:
        raise RuntimeError("Detection service not available")
    
    # 이미지 전처리 (PIL 사용)
    from PIL import Image
    import io
    import numpy as np
    
    # 바이트 데이터를 PIL Image로 변환
    image = Image.open(io.BytesIO(image_data))
    
    # RGB로 변환 (RGBA인 경우)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # PIL Image를 numpy 배열로 변환
    image_rgb = np.array(image)
    
    # 신뢰도 임계값 설정
    conf_threshold = confidence_threshold or detection_service.threshold_config.get("confidence_threshold", 0.5)
    
    # YOLO 모델로 예측 (GUI 기능 완전 비활성화)
    results = detection_service.model.predict(
        source=image_rgb,
        conf=conf_threshold,
        verbose=False,
        show=False,  # GUI 윈도우 표시 비활성화
        save=False,  # 결과 이미지 저장 비활성화
        stream=False  # 스트리밍 비활성화
    )
    
    # 결과 처리
    predictions = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id in detection_service.defect_classes:
                prediction = {
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(class_id),
                    "class_name": detection_service.defect_classes[class_id],
                    "area": float((box[2] - box[0]) * (box[3] - box[1]))
                }
                predictions.append(prediction)
    
    return {
        "predictions": predictions,
        "image_shape": image_rgb.shape,
        "confidence_threshold": conf_threshold,
        "timestamp": datetime.now().isoformat(),
        "model_source": "Hugging Face"
    }


@router.get("/model/info")
async def get_model_info():
    """
    모델 정보 조회
    
    Returns:
        모델 정보 (이름, 타입, 클래스, 설정 등)
    """
    try:
        if detection_service is None:
            raise HTTPException(status_code=503, detail="Detection service not available")
        
        info = await detection_service.get_model_info()
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

