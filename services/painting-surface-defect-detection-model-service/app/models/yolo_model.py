import os
import json
import logging
from typing import Dict, Any, Optional

# OpenCV GUI 기능 비활성화를 위한 환경 변수 설정 (ultralytics import 전에 설정)
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_V4L2'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_1394'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_CAP_ANY'] = '0'

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaintingSurfaceDefectModelLoader:
    """도장 표면 결함 탐지 모델 로더"""
    
    def __init__(self):
        # Hugging Face 저장소 정보
        self.org = os.getenv("HUGGING_FACE_ORG", "23smartfactory")
        self.repo = os.getenv("HUGGING_FACE_REPO", "painting-surface-defect")
        self.model_name = os.getenv("HUGGING_FACE_MODEL_NAME", f"{self.org}/{self.repo}")
        
        # 도장 표면 결함 클래스 정의 (실제 학습 데이터 기준)
        self.defect_classes = {
            0: "dirt",           # 먼지/오염
            1: "runs",           # 흘러내림
            2: "scratch",        # 스크래치
            3: "water_marks"     # 물 자국
        }
        
        self.model = None
        self.model_config = None
        
        logger.info(f"Model loader initialized for {self.model_name}")
    
    def load_yolo_model(self) -> YOLO:
        """Hugging Face에서 YOLO 모델 로드"""
        try:
            logger.info(f"Loading YOLO model from {self.model_name}")
            
            # Hugging Face에서 모델 파일 다운로드
            model_filename = "paint_defect_yolov8m_bbox_best.pt"
            model_path = hf_hub_download(
                repo_id=self.model_name,
                filename=model_filename
            )
            
            # 로컬 파일에서 YOLO 모델 로드
            self.model = YOLO(model_path)
            
            logger.info("YOLO model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model from Hugging Face: {str(e)}")
    
    def load_model_config(self) -> Dict[str, Any]:
        """모델 설정 파일 로드"""
        try:
            config_filename = "model_config.json"
            config_path = hf_hub_download(
                repo_id=self.model_name, 
                filename=config_filename
            )
            
            with open(config_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)
            
            logger.info("Model configuration loaded successfully")
            return self.model_config
            
        except FileNotFoundError:
            logger.warning(f"Model config file '{config_filename}' not found, using empty config")
            return {}
        except Exception as e:
            logger.error(f"Failed to load model config: {str(e)}")
            return {}
    
    def load_class_mapping(self) -> Dict[int, str]:
        """클래스 매핑 파일 로드"""
        try:
            mapping_filename = "class_mapping.json"
            mapping_path = hf_hub_download(
                repo_id=self.model_name, 
                filename=mapping_filename
            )
            
            with open(mapping_path, "r", encoding="utf-8") as f:
                class_mapping = json.load(f)
            
            # 문자열 키를 정수로 변환
            return {int(k): v for k, v in class_mapping.items()}
            
        except FileNotFoundError:
            logger.warning(f"Class mapping file '{mapping_filename}' not found, using default mapping")
            return self.defect_classes
        except Exception as e:
            logger.error(f"Failed to load class mapping: {str(e)}")
            return self.defect_classes
    
    def load_threshold_config(self) -> Dict[str, float]:
        """임계값 설정 파일 로드"""
        try:
            threshold_filename = "thresholds.json"
            threshold_path = hf_hub_download(
                repo_id=self.model_name, 
                filename=threshold_filename
            )
            
            with open(threshold_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            
            logger.info("Threshold configuration loaded successfully")
            return thresholds
            
        except FileNotFoundError:
            logger.warning(f"Threshold file '{threshold_filename}' not found, using empty thresholds")
            return {}
        except Exception as e:
            logger.error(f"Failed to load threshold config: {str(e)}")
            return {}
    

    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            "model_name": self.model_name,
            "model_type": "YOLO",
            "org": self.org,
            "repo": self.repo,
            "defect_classes": self.defect_classes,
            "total_classes": len(self.defect_classes),
            "model_loaded": self.model is not None,
            "config_loaded": self.model_config is not None
        }
        
        if self.model_config:
            info.update(self.model_config)
        
        return info
    
    def __repr__(self) -> str:
        """문자열 표현"""
        return f"PaintingSurfaceDefectModelLoader(org='{self.org}', repo='{self.repo}', model_name='{self.model_name}')"
    
    def validate_model(self) -> bool:
        """모델 유효성 검사"""
        try:
            if self.model is None:
                return False
            
            # 간단한 테스트 이미지로 모델 검증
            import numpy as np
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 예측 테스트 (GUI 기능 비활성화)
            results = self.model.predict(
                source=test_image,
                conf=0.1,
                verbose=False,
                show=False,  # GUI 윈도우 표시 비활성화
                save=False,  # 결과 이미지 저장 비활성화
                stream=False  # 스트리밍 비활성화
            )
            
            logger.info("Model validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False
