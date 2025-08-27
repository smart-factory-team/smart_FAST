import os
import tempfile
import logging
from typing import List, Dict, Any, Optional
import base64
import io
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class InferenceService:
    """추론 서비스 클래스"""
    
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
    
    def _save_base64_image(self, base64_str: str) -> str:
        """Base64 이미지를 임시 파일로 저장"""
        try:
            # Base64 헤더 제거 (data:image/jpeg;base64, 부분)
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            # Base64 디코딩
            image_data = base64.b64decode(base64_str)
            
            # PIL로 이미지 로드
            image = Image.open(io.BytesIO(image_data))
            
            # 임시 파일로 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            image.save(temp_file.name, 'JPEG')
            temp_file.close()
            
            logger.info(f"Base64 이미지를 임시 파일로 저장: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Base64 이미지 저장 실패: {e}")
            raise ValueError(f"Base64 이미지 처리 실패: {e}")
    
    def _cleanup_temp_file(self, file_path: str):
        """임시 파일 정리"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"임시 파일 삭제: {file_path}")
        except Exception as e:
            logger.warning(f"임시 파일 삭제 실패: {e}")
    
    def predict_single_image(self, image_input: str, input_type: str = "file_path") -> Dict[str, Any]:
        """단일 이미지 예측
        
        Args:
            image_input: 이미지 경로 또는 Base64 문자열
            input_type: "file_path" 또는 "base64"
        """
        temp_file_path = None
        
        try:
            # 입력 타입에 따라 처리
            if input_type == "base64":
                image_path = self._save_base64_image(image_input)
                temp_file_path = image_path
            else:
                image_path = image_input
                if not os.path.exists(image_path):
                    raise ValueError(f"이미지 파일이 존재하지 않습니다: {image_path}")
            
            # YOLO 모델로 구멍 탐지
            detections = self.yolo_model.detect_holes(image_path)
            
            # 결과 정리
            result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "image_info": {
                    "input_type": input_type,
                    "processed": True
                },
                "detections": {
                    "total_count": len(detections),
                    "holes": detections
                },
                "categories_detected": list(set([det['category_id'] for det in detections])),
                "category_names_detected": list(set([det['category_name'] for det in detections]))
            }
            
            logger.info(f"단일 이미지 예측 완료: {len(detections)}개 구멍 탐지")
            return result
            
        except Exception as e:
            logger.error(f"단일 이미지 예측 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        finally:
            # 임시 파일 정리
            if temp_file_path:
                self._cleanup_temp_file(temp_file_path)
    
    def predict_multiple_images(self, image_inputs: List[Dict[str, str]]) -> Dict[str, Any]:
        """다중 이미지 예측 (품질 검사)
        
        Args:
            image_inputs: [{"image": "base64 or path", "type": "base64 or file_path", "name": "optional"}]
        """
        temp_files = []
        
        try:
            # 각 이미지별 탐지 결과
            detections_by_image = {}
            failed_images = []
            processed_count = 0
            
            for idx, image_input in enumerate(image_inputs):
                image_data = image_input.get('image')
                input_type = image_input.get('type', 'file_path')
                image_name = image_input.get('name', f'image_{idx}')
                
                try:
                    # 단일 이미지 예측
                    result = self.predict_single_image(image_data, input_type)
                    
                    if result['success']:
                        detections_by_image[image_name] = result['detections']['holes']
                        processed_count += 1
                    else:
                        failed_images.append({
                            'name': image_name,
                            'error': result.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    failed_images.append({
                        'name': image_name,
                        'error': str(e)
                    })
            
            # 적응형 임계값 적용
            quality_result = self.yolo_model.apply_adaptive_thresholds(detections_by_image)
            
            # 최종 결과
            result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "processing_summary": {
                    "total_images": len(image_inputs),
                    "processed_images": processed_count,
                    "failed_images": len(failed_images),
                    "failed_details": failed_images
                },
                "quality_inspection": quality_result,
                "detailed_detections": detections_by_image
            }
            
            logger.info(f"다중 이미지 예측 완료: {processed_count}/{len(image_inputs)}개 처리, 품질상태: {quality_result.get('quality_status', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"다중 이미지 예측 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        finally:
            # 모든 임시 파일 정리
            for temp_file in temp_files:
                self._cleanup_temp_file(temp_file)
    
    def predict_inspection_batch(self, inspection_id: str, images: List[Dict[str, str]]) -> Dict[str, Any]:
        """검사 배치 예측 (21장 이미지 세트)
        
        Args:
            inspection_id: 검사 ID
            images: 21장의 이미지 리스트
        """
        try:
            # 다중 이미지 예측 수행
            result = self.predict_multiple_images(images)
            
            if result['success']:
                # inspection_id 정보 추가
                result['inspection_info'] = {
                    'inspection_id': inspection_id,
                    'expected_images': 21,
                    'actual_images': len(images),
                    'is_complete_dataset': len(images) == 21
                }
                
                # 품질 검사 결과 요약
                quality = result.get('quality_inspection', {})
                result['final_judgment'] = {
                    'inspection_id': inspection_id,
                    'quality_status': quality.get('quality_status', 'Unknown'),
                    'is_complete': quality.get('is_complete', False),
                    'missing_holes': quality.get('missing_category_names', []),
                    'recommendation': 'Pass' if quality.get('is_complete', False) else 'Reject'
                }
            
            logger.info(f"검사 배치 예측 완료: inspection_id={inspection_id}, 결과={result.get('final_judgment', {}).get('recommendation', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"검사 배치 예측 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "inspection_id": inspection_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_service_info(self) -> Dict[str, Any]:
        """서비스 정보 반환"""
        return {
            "service_name": "Press Defect Detection Inference Service",
            "model_info": self.yolo_model.get_model_info() if self.yolo_model else None,
            "supported_formats": ["JPEG", "PNG", "JPG"],
            "input_methods": ["file_upload", "base64"],
            "features": [
                "single_image_detection",
                "multi_image_quality_inspection", 
                "adaptive_threshold_evaluation",
                "batch_processing"
            ]
        }