import base64
import io
from typing import List, Dict, Optional, Tuple
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import AzureError
import asyncio
from PIL import Image

from config.settings import settings
from utils.logger import simulator_logger

class AzureStorageService:
    """Azure Blob Storage 서비스"""
    
    def __init__(self):
        self.blob_service_client = None
        self.container_name = settings.azure_container_name
        self.press_defect_path = settings.azure_press_defect_path
        self.connection_tested = False
        
    async def initialize(self) -> bool:
        """Azure Storage 클라이언트 초기화"""
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                settings.azure_connection_string
            )
            
            # 연결 테스트
            success = await self.test_connection()
            if success:
                self.connection_tested = True
                simulator_logger.log_azure_connection(True, "초기화 완료")
            else:
                simulator_logger.log_azure_connection(False, "초기화 실패")
            
            return success
            
        except Exception as e:
            simulator_logger.log_azure_connection(False, f"초기화 중 오류: {str(e)}")
            return False
    
    async def test_connection(self) -> bool:
        """Azure Storage 연결 테스트"""
        try:
            if not self.blob_service_client:
                return False
            
            # 컨테이너 존재 확인
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            # 컨테이너 속성 조회 (연결 테스트)
            await asyncio.get_event_loop().run_in_executor(
                None, container_client.get_container_properties
            )
            
            simulator_logger.log_azure_connection(True, f"컨테이너 '{self.container_name}' 접근 성공")
            return True
            
        except AzureError as e:
            simulator_logger.log_azure_connection(False, f"Azure 오류: {str(e)}")
            return False
        except Exception as e:
            simulator_logger.log_azure_connection(False, f"연결 테스트 실패: {str(e)}")
            return False
    
    async def get_inspection_list(self) -> List[str]:
        """사용 가능한 inspection 폴더 목록 조회"""
        try:
            if not self.blob_service_client:
                raise Exception("Azure Storage 클라이언트가 초기화되지 않았습니다.")
            
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            # press-defect/ 경로의 모든 blob 조회
            blob_list = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: list(container_client.list_blobs(name_starts_with=f"{self.press_defect_path}/"))
            )
            
            # inspection 폴더 추출
            inspection_folders = set()
            for blob in blob_list:
                blob_path = blob.name
                # press-defect/inspection_001/image.jpg 형태에서 inspection_001 추출
                if blob_path.startswith(f"{self.press_defect_path}/inspection_"):
                    parts = blob_path.split('/')
                    if len(parts) >= 3:
                        inspection_folder = parts[1]  # inspection_001
                        inspection_folders.add(inspection_folder)
            
            inspection_list = sorted(list(inspection_folders))
            simulator_logger.logger.info(f"발견된 inspection 폴더: {len(inspection_list)}개")
            
            return inspection_list
            
        except Exception as e:
            simulator_logger.logger.error(f"inspection 목록 조회 실패: {str(e)}")
            return []
    
    async def get_inspection_images(self, inspection_id: int) -> List[Dict[str, str]]:
        """특정 inspection의 이미지 목록 조회"""
        try:
            inspection_folder = f"inspection_{inspection_id:03d}"
            folder_path = f"{self.press_defect_path}/{inspection_folder}"
            
            if not self.blob_service_client:
                raise Exception("Azure Storage 클라이언트가 초기화되지 않았습니다.")
            
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            # 해당 폴더의 이미지 파일들 조회
            blob_list = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(container_client.list_blobs(name_starts_with=f"{folder_path}/"))
            )
            
            images = []
            for blob in blob_list:
                blob_name = blob.name
                if blob_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # 파일명에서 카메라 정보 추출
                    file_name = blob_name.split('/')[-1]  # image.jpg
                    base_name = file_name.rsplit('.', 1)[0]  # image
                    
                    images.append({
                        'blob_name': blob_name,
                        'file_name': file_name,
                        'camera_name': base_name,
                        'size': blob.size
                    })
            
            # 파일명으로 정렬
            images.sort(key=lambda x: x['file_name'])
            
            simulator_logger.logger.info(f"{inspection_folder}: {len(images)}장 이미지 발견")
            return images
            
        except Exception as e:
            simulator_logger.logger.error(f"inspection_{inspection_id:03d} 이미지 목록 조회 실패: {str(e)}")
            return []
    
    async def download_image_as_base64(self, blob_name: str) -> Optional[str]:
        """이미지를 Base64로 다운로드"""
        try:
            if not self.blob_service_client:
                raise Exception("Azure Storage 클라이언트가 초기화되지 않았습니다.")
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # 이미지 다운로드
            blob_data = await asyncio.get_event_loop().run_in_executor(
                None, blob_client.download_blob
            )
            
            image_bytes = blob_data.readall()
            
            # Base64 인코딩
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            # MIME 타입 추가
            if blob_name.lower().endswith('.png'):
                base64_with_header = f"data:image/png;base64,{base64_string}"
            else:
                base64_with_header = f"data:image/jpeg;base64,{base64_string}"
            
            return base64_with_header
            
        except Exception as e:
            simulator_logger.logger.error(f"이미지 다운로드 실패 ({blob_name}): {str(e)}")
            return None
    
    async def download_inspection_images(self, inspection_id: int) -> Tuple[bool, List[Dict[str, str]]]:
        """inspection의 모든 이미지를 Base64로 다운로드"""
        try:
            # 이미지 목록 조회
            image_list = await self.get_inspection_images(inspection_id)
            
            if not image_list:
                return False, []
            
            # 각 이미지를 Base64로 변환
            images_data = []
            successful_downloads = 0
            
            for image_info in image_list:
                blob_name = image_info['blob_name']
                base64_data = await self.download_image_as_base64(blob_name)
                
                if base64_data:
                    images_data.append({
                        'image': base64_data,
                        'name': image_info['camera_name']
                    })
                    successful_downloads += 1
                else:
                    simulator_logger.logger.warning(f"이미지 다운로드 실패: {blob_name}")
            
            success = successful_downloads > 0
            
            if success:
                simulator_logger.logger.info(
                    f"inspection_{inspection_id:03d}: {successful_downloads}/{len(image_list)}장 다운로드 완료"
                )
            else:
                simulator_logger.logger.error(f"inspection_{inspection_id:03d}: 모든 이미지 다운로드 실패")
            
            return success, images_data
            
        except Exception as e:
            simulator_logger.logger.error(f"inspection_{inspection_id:03d} 이미지 다운로드 실패: {str(e)}")
            return False, []
    
    async def validate_inspection_completeness(self, inspection_id: int) -> Dict[str, any]:
        """inspection 데이터 완성도 검사"""
        try:
            image_list = await self.get_inspection_images(inspection_id)
            
            expected_count = 21  # 예상 이미지 수
            actual_count = len(image_list)
            
            # 카메라별 파일 확인
            camera_files = {}
            for img in image_list:
                camera_name = img['camera_name']
                camera_files[camera_name] = img
            
            validation_result = {
                'inspection_id': inspection_id,
                'expected_count': expected_count,
                'actual_count': actual_count,
                'is_complete': actual_count >= expected_count,
                'missing_count': max(0, expected_count - actual_count),
                'camera_files': camera_files,
                'camera_count': len(camera_files)
            }
            
            return validation_result
            
        except Exception as e:
            simulator_logger.logger.error(f"inspection_{inspection_id:03d} 완성도 검사 실패: {str(e)}")
            return {
                'inspection_id': inspection_id,
                'expected_count': 21,
                'actual_count': 0,
                'is_complete': False,
                'error': str(e)
            }
    
    def get_service_status(self) -> Dict[str, any]:
        """Azure Storage 서비스 상태 반환"""
        return {
            'service_name': 'Azure Blob Storage',
            'connected': self.connection_tested,
            'container_name': self.container_name,
            'press_defect_path': self.press_defect_path,
            'client_initialized': self.blob_service_client is not None
        }

# 전역 Azure Storage 서비스 인스턴스
azure_storage_service = AzureStorageService()