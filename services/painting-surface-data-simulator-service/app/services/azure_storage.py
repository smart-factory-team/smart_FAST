from typing import List, Dict, Any, Optional
import asyncio
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import AzureError, ClientAuthenticationError
from app.config.settings import settings


class AzureStorageService:
    def __init__(self):
        self.connection_string = settings.azure_connection_string
        self.container_name = settings.azure_container_name
        self.client = None
        
        # 도장 표면 이미지 처리를 위한 인덱스
        self.image_index = 0

    async def connect(self):
        """Azure Blob Storage 클라이언트 초기화"""
        if not self.connection_string:
            raise ValueError("Azure connection string이 설정되지 않았습니다.")

        try:
            self.client = BlobServiceClient.from_connection_string(
                self.connection_string)
            
            # 연결 테스트
            await self._test_connection()
            print(f"✅ Azure Blob Storage 연결 성공: {self.container_name}")
            
        except ClientAuthenticationError as e:
            print(f"❌ Azure Storage 인증 실패: {e}")
            print(f"   연결 문자열을 확인해주세요: {self.connection_string[:50]}...")
            raise
        except Exception as e:
            print(f"❌ Azure Storage 연결 실패: {e}")
            raise

    async def _test_connection(self):
        """연결 테스트 - 컨테이너 존재 여부 확인"""
        try:
            container_client = self.client.get_container_client(self.container_name)
            properties = await container_client.get_container_properties()
            print(f"📦 컨테이너 확인: {properties.name}")
            # created_on 속성이 없을 수 있으므로 안전하게 처리
            if hasattr(properties, 'created_on'):
                print(f"   생성일: {properties.created_on}")
            else:
                print("   생성일: 알 수 없음")
        except Exception as e:
            print(f"⚠️ 컨테이너 접근 실패: {e}")
            raise

    async def disconnect(self):
        """연결 종료"""
        if self.client:
            await self.client.close()

    async def list_data_files(self) -> List[str]:
        """데이터 파일 목록 조회"""
        try:
            if not self.client:
                await self.connect()
                
            container_client = self.client.get_container_client(
                self.container_name)
            blob_list = []

            # 도장 표면 이미지 폴더 검색
            prefix = f"{settings.painting_data_folder}/"
            print(f"🔍 검색 중: {prefix}")

            async for blob in container_client.list_blobs(name_starts_with=prefix):
                if blob.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    blob_list.append(blob.name)
                    print(f"📁 발견된 파일: {blob.name}")

            print(f"📊 총 {len(blob_list)}개의 이미지 파일 발견")
            return sorted(blob_list)
            
        except ClientAuthenticationError as e:
            print(f"❌ 인증 오류로 인한 파일 목록 조회 실패: {e}")
            print("   Azure Storage 계정 키와 연결 문자열을 확인해주세요.")
            return []
        except Exception as e:
            print(f"❌ 도장 표면 이미지 파일 목록 조회 실패: {e}")
            return []

    async def read_image_data(self, blob_name: str) -> Optional[bytes]:
        """이미지 파일 읽기"""
        try:
            if not self.client:
                await self.connect()
                
            blob_client = self.client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            # 비동기로 blob 데이터 다운로드
            blob_data = await blob_client.download_blob()
            content = await blob_data.readall()

            print(f"📁 이미지 읽기 성공: {blob_name} ({len(content)} bytes)")
            return content

        except Exception as e:
            print(f"❌ 이미지 읽기 실패 ({blob_name}): {e}")
            return None

    async def get_recent_data_files(self) -> Dict[str, List[str]]:
        """최근 N시간 내 도장 표면 이미지 파일들 반환"""
        try:
            all_files = await self.list_data_files()

            # painting-surface 전용
            process_files = {
                "painting-surface": all_files  # 모든 파일이 도장 표면 이미지
            }

            return process_files

        except Exception as e:
            print(f"❌ 최근 도장 표면 이미지 파일 조회 실패: {e}")
            return {"painting-surface": []}

    async def simulate_painting_surface_data(self) -> Optional[Dict[str, List[str]]]:
        """도장 표면 결함 감지 이미지 데이터 시뮬레이션"""
        try:
            # 도장 표면 이미지 파일 목록 조회
            image_files = await self.list_data_files()
            
            if not image_files:
                print("⚠️ 도장 표면 이미지 파일이 없습니다.")
                return None
            
            # 순차적으로 이미지 처리 (배치 크기만큼)
            batch_size = min(settings.batch_size, len(image_files))
            start_idx = self.image_index % len(image_files)
            end_idx = min(start_idx + batch_size, len(image_files))
            
            batch_files = image_files[start_idx:end_idx]
            self.image_index = (self.image_index + batch_size) % len(image_files)
            
            print(f"📸 이미지 배치 처리: {len(batch_files)}개 (전체: {len(image_files)}개)")
            
            return {
                "images": batch_files,
                "total_images": len(image_files),
                "batch_size": batch_size,
                "current_batch": batch_files
            }
            
        except Exception as e:
            print(f"❌ 도장 표면 데이터 시뮬레이션 실패: {e}")
            return None


# 글로벌 Azure Storage 서비스 인스턴스
azure_storage = AzureStorageService()
