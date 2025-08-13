from typing import List, Optional
from azure.storage.blob.aio import BlobServiceClient
import logging
import ssl
import certifi

logger = logging.getLogger(__name__)

class AzureStorageService:
    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self.client = None

    async def connect(self):
        """Azure Blob Storage 클라이언트 초기화 (SSL 설정 포함)"""
        try:
            # SSL 컨텍스트 설정
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            # BlobServiceClient 생성 시 SSL 설정 적용
            self.client = BlobServiceClient.from_connection_string(
                self.connection_string,
                ssl_context=ssl_context  # SSL 컨텍스트 추가
            )

            logger.info(f"Azure Blob Storage 연결 성공: {self.container_name}")

        except Exception as e:
            logger.error(f"Azure 연결 실패: {e}")
            raise

    async def disconnect(self):
        """연결 종료"""
        if self.client:
            await self.client.close()

    async def list_image_files(self, prefix: str = "") -> List[str]:
        """이미지 파일 목록 조회 (로그 강화)"""
        try:
            logger.info(f"🔍 파일 검색 시작 - 컨테이너: '{self.container_name}', prefix: '{prefix}'")

            container_client = self.client.get_container_client(self.container_name)
            blob_list = []
            all_files = []

            # 모든 파일 수집
            file_count = 0
            async for blob in container_client.list_blobs(name_starts_with=prefix):
                file_count += 1
                all_files.append(blob.name)

                # 처음 10개 파일명 출력
                if file_count <= 10:
                    logger.info(f"📁 발견된 파일 {file_count}: '{blob.name}'")

                # 이미지 파일 필터링
                if blob.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    blob_list.append(blob.name)
                else:
                    # 확장자 확인
                    file_ext = blob.name.split('.')[-1] if '.' in blob.name else 'no extension'
                    logger.info(f"❌ 스킵 (확장자: {file_ext}): {blob.name}")  # debug → info로 변경

            # 결과 요약 로그 추가
            logger.info(f"📊 총 {file_count}개 파일 발견")
            logger.info(f"🖼️  이미지 파일: {len(blob_list)}개")

            if len(blob_list) == 0:
                logger.warning("⚠️  이미지 파일이 없습니다!")

            return sorted(blob_list)

        except Exception as e:
            logger.error(f"이미지 파일 목록 조회 실패: {e}")
            return []


    async def download_image(self, blob_name: str) -> Optional[bytes]:
        """이미지 파일 다운로드"""
        try:
            blob_client = self.client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            # 파일 존재 확인
            logger.info(f"🔍 파일 존재 확인 중...")
            properties = await blob_client.get_blob_properties()
            logger.info(f"✅ 파일 확인됨: {properties.size} bytes")

            # 다운로드 실행
            logger.info(f"📥 다운로드 실행 중...")
            blob_data = await blob_client.download_blob()
            content = await blob_data.readall()

            logger.info(f"✅ 다운로드 완료: {blob_name} ({len(content)} bytes)")
            return content
        except Exception as e:
            logger.error(f"이미지 다운로드 실패 ({blob_name}): {e}")
            return None