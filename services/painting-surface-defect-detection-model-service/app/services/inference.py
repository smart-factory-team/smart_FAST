import asyncio
import os
from typing import Dict, List, Optional, Any
import numpy as np

# OpenCV GUI 기능 비활성화를 위한 환경 변수 설정
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_V4L2'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_1394'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_CAP_ANY'] = '0'

from PIL import Image
import io
import base64
from datetime import datetime
import logging

# YOLO 관련 라이브러리
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics library not found. Please install: pip install ultralytics")

# 모델 로더 import
from app.models.yolo_model import PaintingSurfaceDefectModelLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaintingSurfaceDefectDetectionService:
    """도장 표면 결함 탐지 서비스"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_loader = PaintingSurfaceDefectModelLoader()
        self.model_config = None
        self.threshold_config = None
        self.defect_classes = None
        
        logger.info("Service initialized")
    
    async def load_model(self):
        """Hugging Face에서 모델 및 설정 로드"""
        try:
            logger.info("Loading model and configurations from Hugging Face...")
            
            # YOLO 모델 로드
            self.model = self.model_loader.load_yolo_model()
            
            # 모델 설정 로드
            self.model_config = self.model_loader.load_model_config()
            
            # 임계값 설정 로드
            self.threshold_config = self.model_loader.load_threshold_config()
            
            # 클래스 매핑 로드
            self.defect_classes = self.model_loader.load_class_mapping()
            
            # 모델 유효성 검사
            if not self.model_loader.validate_model():
                raise RuntimeError("Model validation failed")
            
            self.model_loaded = True
            logger.info("Model and configurations loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def is_model_loaded(self) -> bool:
        """모델 로딩 상태 확인"""
        return self.model_loaded and self.model is not None
    
    async def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """이미지 전처리 (PIL 사용)"""
        try:
            # 바이트 데이터를 PIL Image로 변환
            image = Image.open(io.BytesIO(image_data))
            
            # RGB로 변환 (RGBA인 경우)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # PIL Image를 numpy 배열로 변환
            image_array = np.array(image)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise e
    
    async def predict(self, image_data: bytes, confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """도장 표면 결함 탐지 예측"""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # 이미지 전처리
            image = await self.preprocess_image(image_data)
            
            # 신뢰도 임계값 설정
            conf_threshold = confidence_threshold or self.threshold_config.get("confidence_threshold", 0.5)
            
            # YOLO 모델로 예측 (GUI 기능 완전 비활성화)
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                verbose=False,
                show=False,  # GUI 윈도우 표시 비활성화
                save=False,  # 결과 이미지 저장 비활성화
                stream=False  # 스트리밍 비활성화
            )
            
            predictions = self._process_yolo_results(results[0])
            
            return {
                "predictions": predictions,
                "image_shape": image.shape,
                "confidence_threshold": conf_threshold,
                "timestamp": datetime.now().isoformat(),
                "model_source": "Hugging Face",
                "model_config": self.model_config
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise e
    
    def _process_yolo_results(self, result) -> List[Dict[str, Any]]:
        """YOLO 결과 처리"""
        predictions = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if class_id in self.defect_classes:
                    prediction = {
                        "bbox": box.tolist(),  # [x1, y1, x2, y2]
                        "confidence": float(conf),
                        "class_id": int(class_id),
                        "class_name": self.defect_classes[class_id],
                        "area": float((box[2] - box[0]) * (box[3] - box[1]))
                    }
                    predictions.append(prediction)
        
        return predictions
    
    async def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if not self.is_model_loaded():
            return {"error": "Model not loaded"}
        
        info = self.model_loader.get_model_info()
        info.update({
            "model_loaded": self.model_loaded,
            "threshold_config": self.threshold_config,
            "timestamp": datetime.now().isoformat()
        })
        
        return info
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.model is not None:
                del self.model
            
            # CUDA 캐시 정리 (가능한 경우)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            self.model_loaded = False
            self.model_config = None
            self.threshold_config = None
            self.defect_classes = None
            
            logger.info("Model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    async def predict_batch(self, image_data_list: List[bytes], confidence_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """배치 예측"""
        results = []
        for image_data in image_data_list:
            try:
                result = await self.predict(image_data, confidence_threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch prediction: {str(e)}")
                results.append({"error": str(e)})
        
        return results
