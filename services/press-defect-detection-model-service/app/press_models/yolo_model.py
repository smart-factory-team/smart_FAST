import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging
from typing import List, Dict, Any
import tempfile
import shutil

logger = logging.getLogger(__name__)

class YOLOv7Model:
    """YOLOv7 기반 프레스 구멍 결함 탐지 모델"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.model_loaded = False
        
        # 카테고리 정의
        self.category_names = {
            0: 'AX1', 1: 'BY1', 2: 'CY1', 3: 'DY1', 
            4: 'DY2', 5: 'DY3', 6: 'DY4'
        }
        
        # 최적화된 적응형 임계값 (재현율 개선 버전)
        self.adaptive_thresholds = {
            0: 0.15,  # AX1: 15%
            1: 0.50,  # BY1: 50%
            2: 0.20,  # CY1: 20%
            3: 0.40,  # DY1: 40%
            4: 0.45,  # DY2: 45%
            5: 0.25,  # DY3: 25%
            6: 0.35   # DY4: 35%
        }
        
        # YOLOv7 레포지토리 경로
        self.yolov7_path = None
        self.yolov7_loaded = False
    
    def _setup_yolov7_repo(self):
        """YOLOv7 레포지토리 설정"""
        try:
            # # 현재 디렉토리에서 yolov7 폴더 찾기
            # current_dir = os.getcwd()
            # yolov7_path = os.path.join(current_dir, 'yolov7')

            # 이 파일이 위치한 디렉토리 기준 절대경로 설정
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
            yolov7_path = os.path.join(base_dir, 'yolov7')
            
            if not os.path.exists(yolov7_path):
                logger.error(f"YOLOv7 레포지토리를 찾을 수 없습니다: {yolov7_path}")
                logger.info("다음 명령어로 YOLOv7를 클론하세요: git clone https://github.com/WongKinYiu/yolov7.git")
                return False
            
            # # YOLOv7 경로를 Python path에 추가
            # if yolov7_path not in sys.path:
            #     sys.path.insert(0, yolov7_path)
            #     logger.info(f"YOLOv7 경로를 sys.path 맨 앞에 추가: {yolov7_path}")
            
            sys.path.insert(0, yolov7_path)
            logger.info(f"YOLOv7 경로를 sys.path 맨 앞에 추가: {yolov7_path}")
            
            self.yolov7_path = yolov7_path
            logger.info(f"YOLOv7 경로 설정 완료: {yolov7_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"YOLOv7 레포지토리 설정 실패: {e}")
            return False
    
    def _download_model_from_huggingface(self):
        """허깅페이스에서 모델 다운로드"""
        try:
            logger.info("허깅페이스에서 모델 다운로드 중...")
            
            # 모델 파일 다운로드
            model_path = hf_hub_download(
                repo_id="23smartfactory/press-defect-detection-model",
                filename="press_hole_yolov7_best.pt",
                cache_dir="./model_cache"  # 로컬 캐시 디렉토리
            )
            
            logger.info(f"모델 다운로드 완료: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"모델 다운로드 실패: {e}")
            raise
    
    def _load_yolov7_dependencies(self):
        """YOLOv7 의존성 로딩"""
        try:
            logger.info("YOLOv7 의존성 로딩 중...")
            
            # YOLOv7 레포지토리 설정
            if not self._setup_yolov7_repo():
                return False
            
            # 디바이스 설정
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"사용 디바이스: {self.device}")
            
            # YOLOv7 모듈 import 시도
            try:
                from models.experimental import attempt_load
                from utils.general import check_img_size, non_max_suppression, scale_coords
                from utils.datasets import letterbox
                
                # 필요한 함수들을 인스턴스 변수로 저장
                self.attempt_load = attempt_load
                self.non_max_suppression = non_max_suppression
                self.scale_coords = scale_coords
                self.letterbox = letterbox
                self.check_img_size = check_img_size
                
                self.yolov7_loaded = True
                logger.info("YOLOv7 의존성 로딩 완료")
                return True
                
            except ImportError as e:
                logger.error(f"YOLOv7 모듈 import 실패: {e}")
                return False
            
        except Exception as e:
            logger.error(f"YOLOv7 의존성 로딩 실패: {e}")
            return False

    async def load_model(self):
        """모델 비동기 로딩"""
        try:
            logger.info("🤖 AI 모델 로딩 시작...")
            
            # 1. YOLOv7 의존성 로딩
            if not self._load_yolov7_dependencies():
                raise Exception("YOLOv7 의존성 로딩 실패")
            
            # 2. 허깅페이스에서 모델 다운로드
            model_path = self._download_model_from_huggingface()
            
            # 3. YOLOv7의 attempt_load 함수 사용
            logger.info("YOLOv7 모델 로딩 중...")
            
            # YOLOv7의 attempt_load 함수 사용
            self.model = self.attempt_load(model_path, map_location=self.device)
            
            # 이미지 크기 체크
            self.img_size = self.check_img_size(self.img_size, s=int(self.model.stride.max()))
            
            # 모델을 평가 모드로 설정
            self.model.eval()
            
            self.model_loaded = True
            logger.info("✅ AI 모델 로딩 완료!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ AI 모델 로딩 실패: {e}")
            self.model_loaded = False
            raise
    
    def _preprocess_image(self, image_path: str):
        """이미지 전처리 (YOLOv7 방식)"""
        try:
            if not self.yolov7_loaded:
                raise Exception("YOLOv7 의존성이 로딩되지 않았습니다.")
            
            # 이미지 로딩
            img0 = cv2.imread(image_path)
            if img0 is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            # YOLOv7의 letterbox 전처리 사용
            img = self.letterbox(img0, self.img_size, stride=int(self.model.stride.max()))[0]
            
            # BGR to RGB, HWC to CHW
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            
            # 정규화 및 텐서 변환
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            return img, img0
            
        except Exception as e:
            logger.error(f"이미지 전처리 실패: {e}")
            raise
    
    def _postprocess_detections(self, pred, img, img0):
        """탐지 결과 후처리 (YOLOv7 방식)"""
        try:
            if not self.yolov7_loaded:
                raise Exception("YOLOv7 의존성이 로딩되지 않았습니다.")
            
            # NMS 적용
            pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
            
            results = []
            
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # 좌표를 원본 이미지 크기로 변환
                    det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    
                    # 결과 저장
                    for *xyxy, conf, cls in reversed(det):
                        category_id = int(cls)
                        confidence = float(conf)
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        results.append({
                            'category_id': category_id,
                            'category_name': self.category_names.get(category_id, f'unknown_{category_id}'),
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'bbox_center': [(x1+x2)/2, (y1+y2)/2]
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"후처리 실패: {e}")
            return []
    
    def detect_holes(self, image_path: str) -> List[Dict[str, Any]]:
        """단일 이미지에서 구멍 탐지"""
        if not self.model_loaded or self.model is None:
            raise Exception("모델이 로딩되지 않았습니다.")
        
        try:
            # 전처리
            img, img0 = self._preprocess_image(image_path)
            
            # 추론
            with torch.no_grad():
                pred = self.model(img, augment=False)[0]
            
            # 후처리
            detections = self._postprocess_detections(pred, img, img0)
            
            logger.info(f"탐지 결과: {len(detections)}개 구멍 발견")
            return detections
            
        except Exception as e:
            logger.error(f"구멍 탐지 실패: {e}")
            raise
    
    def apply_adaptive_thresholds(self, detections_by_image: Dict) -> Dict:
        """적응형 임계값을 적용한 품질 검사"""
        try:
            total_images = len(detections_by_image)
            if total_images == 0:
                return {'is_complete': False, 'error': '처리된 이미지가 없습니다.'}
            
            # 카테고리별 투표 수 계산
            category_vote_count = {cat_id: 0 for cat_id in range(7)}
            
            for image_key, detections in detections_by_image.items():
                detected_categories = set()
                for det in detections:
                    if det['confidence'] >= 0.5:  # 최소 신뢰도
                        detected_categories.add(det['category_id'])
                
                for cat_id in detected_categories:
                    category_vote_count[cat_id] += 1
            
            # 적응형 임계값 적용
            existing_categories = set()
            missing_categories = []
            category_results = {}
            
            for cat_id in range(7):
                threshold_ratio = self.adaptive_thresholds[cat_id]
                required_votes = max(1, int(total_images * threshold_ratio))
                actual_votes = category_vote_count[cat_id]
                vote_ratio = actual_votes / total_images
                
                is_existing = actual_votes >= required_votes
                
                if is_existing:
                    existing_categories.add(cat_id)
                else:
                    missing_categories.append(cat_id)
                
                category_results[cat_id] = {
                    'category_name': self.category_names[cat_id],
                    'threshold_ratio': threshold_ratio,
                    'required_votes': required_votes,
                    'actual_votes': actual_votes,
                    'vote_ratio': vote_ratio,
                    'is_existing': is_existing
                }
            
            # 최종 판정
            is_complete = len(existing_categories) == 7
            quality_status = "정상품" if is_complete else "결함품"
            
            return {
                'is_complete': is_complete,
                'quality_status': quality_status,
                'existing_categories': sorted(list(existing_categories)),
                'missing_categories': sorted(missing_categories),
                'missing_category_names': [self.category_names[cat_id] for cat_id in missing_categories],
                'category_results': category_results,
                'processed_images': total_images
            }
            
        except Exception as e:
            logger.error(f"적응형 임계값 적용 실패: {e}")
            return {'is_complete': False, 'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_loaded': self.model_loaded,
            'yolov7_loaded': self.yolov7_loaded,
            'yolov7_path': self.yolov7_path,
            'device': str(self.device) if self.device else None,
            'img_size': self.img_size,
            'conf_thres': self.conf_thres,
            'iou_thres': self.iou_thres,
            'categories': self.category_names,
            'adaptive_thresholds': {
                self.category_names[cat_id]: f"{threshold*100:.0f}%" 
                for cat_id, threshold in self.adaptive_thresholds.items()
            }
        }