import pandas as pd
from azure.storage.blob.aio import BlobServiceClient
from typing import Optional, Tuple, List
from io import StringIO

from app.config.settings import settings
from app.utils.logger import system_log


class AzureStorageService:
    """Azure Blob Storage에서 CSV 데이터를 일정 단위로 가져오는 서비스"""

    def __init__(self):
        self.blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME
        self.folder_path = settings.PRESS_FAULT_FOLDER.rstrip("/") + "/"

        self.is_connected = False
        # 현재 처리 중인 파일 상태
        self.current_file_name: Optional[str] = None
        self.current_row_index: int = 0
        self.row_per_minute: int = 600  # 1분당 600 행

        system_log.info(
            f"Azure Storage Service 초기화 완료 - Container: {self.container_name}(폴더: {self.folder_path})"
        )
        if self.folder_path:
            system_log.info(f"폴더 경로: {self.folder_path}")

    async def connect(self) -> bool:
        """
        Azure Blob Storage에 연결
        Returns:
            bool: 연결 성공 여부
        """
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            # 컨테이너 존재 여부 확인
            await container_client.get_container_properties()
            system_log.info("Azure Storage 연결 성공")
            self.is_connected = True
            return True
        except Exception as e:
            system_log.error(f"❌Azure Storage 연결 실패: {str(e)}")
            self.is_connected = False
            return False

    async def get_csv_files_list(self) -> List[str]:
        """
        폴더 내 CSV 파일 목록을 최신순 반환
        Returns:
            List[str]: CSV 파일명 리스트
        """
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            csv_files = []

            async for blob in container_client.list_blobs(
                name_starts_with=self.folder_path
            ):
                if blob.name.lower().endswith(".csv"):
                    # 하위 폴더는 제외 (현재 폴더의 파일만)
                    relative_path = blob.name[len(self.folder_path) :]
                    if "/" not in relative_path:
                        csv_files.append((blob.name, blob.last_modified))
            # 최신순으로 정렬하여 파일명만 반환
            sorted_files = sorted(csv_files, key=lambda x: x[1], reverse=True)
            file_names = [file[0] for file in sorted_files]

            system_log.info(f"CSV 파일 목록 조회 완료: {len(file_names)}개 파일")
            return file_names
        except Exception as e:
            system_log.error(f"CSV 파일 목록 조회 실패: {str(e)}")
            return []

    async def load_csv_chunk_from_storage(
        self, file_name: str, start_row: int, num_rows: int
    ) -> Optional[pd.DataFrame]:
        """
        Azure Storage에서 CSV 파일의 특정 행 범위를 직접 로드

        Args:
            file_name: CSV 파일명
            start_row: 시작 행 번호 (0부터 시작, 헤더 제외)
            num_rows: 읽을 행 수

        Returns:
            pd.DataFrame: 로드된 데이터 또는 None (실패 시)
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=file_name
            )

            download_stream = await blob_client.download_blob()
            csv_data = await download_stream.readall()
            csv_text = csv_data.decode("utf-8")

            df = pd.read_csv(
                StringIO(csv_text), skiprows=range(1, start_row + 1), nrows=num_rows
            )
            # 필수 컬럼 확인
            required_columns = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                system_log.error(f"필수 컬럼 누락: {missing_columns}")
                return None

            return df

        except Exception as e:
            system_log.error(f"CSV 청크 로드 실패 ({file_name}): {str(e)}")
            return None

    async def get_next_minute_data(self, _is_retry: bool = False) -> Optional[Tuple[pd.DataFrame, str, bool]]:
        """
        다음 1분치 데이터를 반환
        Args:
            _is_retry (bool): 재귀 호출 여부를 나타내는 내부 플래그

        Returns:
            Tuple[pd.DataFrame, str, bool]: 데이터, 파일명, 파일 끝 여부
            or None
        """
        # 현재 처리 중인 파일이 없으면 최신 파일 선택
        if self.current_file_name is None:
            if _is_retry:
                system_log.warning("재귀 호출에도 불구하고 처리할 새 파일을 찾지 못했습니다.")
                return None

            csv_files = await self.get_csv_files_list()
            if not csv_files:
                system_log.warning("CSV 파일이 없습니다.")
                return None

            self.current_file_name = csv_files[0]  # 최신 파일
            self.current_row_index = 0
            system_log.info(f"새 파일 처리 시작: {self.current_file_name}")

        # Azure Storage에서 현재 행부터 1분치 데이터 로드
        chunk_data = await self.load_csv_chunk_from_storage(
            file_name=self.current_file_name,
            start_row=self.current_row_index,
            num_rows=self.row_per_minute,
        )

        if chunk_data is None:
            system_log.error(f"파일 '{self.current_file_name}'에서 데이터 로드 실패")
            return None

        # 데이터가 예상보다 적으면 파일 끝에 도달한 것
        is_end_of_file = len(chunk_data) < self.row_per_minute

        if len(chunk_data) == 0:
            system_log.info(f"파일 '{self.current_file_name}' 처리 완료")
            # 다음 파일로 이동
            self._move_to_next_file()
            return await self.get_next_minute_data(_is_retry=True)  # 재귀 호출로 다음 파일 처리

        # 다음 처리를 위해 인덱스 업데이트
        old_index = self.current_row_index
        self.current_row_index += len(chunk_data)

        system_log.info(
            f"데이터 청크 로드: {self.current_file_name} "
            f"[{old_index}:{self.current_row_index}] ({len(chunk_data)}행)"
        )

        # 파일 끝 도달 시 다음 호출을 위해 초기화
        if is_end_of_file:
            system_log.info(f"파일 '{self.current_file_name}' 끝에 도달")
            self._move_to_next_file()

        return chunk_data, self.current_file_name, is_end_of_file

    def _move_to_next_file(self):
        """다음 파일로 이동 (상태 초기화)"""
        self.current_file_name = None
        self.current_row_index = 0

    def get_current_status(self) -> dict:
        """현재 처리 상태 반환"""
        if self.current_file_name is None:
            return {"status": "no_file_loaded"}

        return {
            "status": "processing",
            "current_file": self.current_file_name,
            "current_index": self.current_row_index,
            "rows_per_chunk": self.row_per_minute,
        }

    def _clear_cache(self):
        """캐시된 DataFrame 정리"""
        self.current_file_name = None
        self.current_row_index = 0

    async def close(self):
        """연결 종료 (선택적)"""
        try:
            if hasattr(self, "blob_service_client") and self.blob_service_client:
                await self.blob_service_client.close()
                system_log.info("Azure Storage 연결 종료 완료")
        except Exception as e:
            system_log.error(f"Azure Storage 연결 종료 중 오류: {str(e)}")
        finally:
            self.is_connected = False


azure_storage = AzureStorageService()
