import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from azure.storage.blob.aio import BlobServiceClient
from app.config.settings import settings
import io


class AzureStorageService:
    def __init__(self):
        self.connection_string = settings.azure_connection_string
        self.container_name = settings.azure_container_name
        self.client = None

        # 순차 처리를 위한 인덱스 관리
        self.current_index = 0
        self.vibration_index = 0
        self.cached_current_df = None
        self.cached_vibration_df = None

    async def connect(self):
        """Azure Blob Storage 클라이언트 초기화"""
        if not self.connection_string:
            raise ValueError("Azure connection string이 설정되지 않았습니다.")

        self.client = BlobServiceClient.from_connection_string(
            self.connection_string)
        print(f"✅ Azure Blob Storage 연결 성공: {self.container_name}")

    async def disconnect(self):
        """연결 종료"""
        if self.client:
            await self.client.close()

    async def list_data_files(self) -> List[str]:
        """데이터 파일 목록 조회"""
        try:
            container_client = self.client.get_container_client(
                self.container_name)
            blob_list = []

            # welding-machine 폴더만 검색
            prefix = f"{settings.welding_data_folder}/"

            async for blob in container_client.list_blobs(name_starts_with=prefix):
                if blob.name.endswith('.csv'):
                    blob_list.append(blob.name)

            return sorted(blob_list)
        except Exception as e:
            print(f"❌ Welding 데이터 파일 목록 조회 실패: {e}")
            return []

    async def read_csv_data(self, blob_name: str) -> Optional[pd.DataFrame]:
        """CSV 파일 읽기"""
        try:
            blob_client = self.client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            # 비동기로 blob 데이터 다운로드
            blob_data = await blob_client.download_blob()
            content = await blob_data.readall()

            # CSV를 DataFrame으로 변환
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))

            print(f"📁 파일 읽기 성공: {blob_name} ({len(df)} rows)")
            return df

        except Exception as e:
            print(f"❌ 파일 읽기 실패 ({blob_name}): {e}")
            return None

    async def get_recent_data_files(self) -> Dict[str, List[str]]:
        """최근 N시간 내 Welding Machine 데이터 파일들 반환"""
        try:
            all_files = await self.list_data_files()

            # welding-machine 전용
            process_files = {
                "welding-machine": all_files  # 모든 파일이 welding 파일
            }

            return process_files

        except Exception as e:
            print(f"❌ 최근 Welding 파일 조회 실패: {e}")
            return {"welding-machine": []}

    async def simulate_real_time_data(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Welding Machine 실시간 데이터 시뮬레이션 (전류 + 진동) - 순차 처리"""
        try:
            # 캐시된 DataFrame이 없으면 로드
            if self.cached_current_df is None or self.cached_vibration_df is None:
                await self._load_dataframes()

            if self.cached_current_df is None or self.cached_vibration_df is None:
                return None

            # 순차적으로 행 선택
            current_row = self.cached_current_df.iloc[self.current_index]
            vibration_row = self.cached_vibration_df.iloc[self.vibration_index]

            # 인덱스 업데이트 (순환)
            self.current_index = (self.current_index +
                                  1) % len(self.cached_current_df)
            self.vibration_index = (
                self.vibration_index + 1) % len(self.cached_vibration_df)

            # 전류 데이터: 2~1025열 추출 (인덱스 1~1024, 0-based) → 1024개
            current_values = current_row.iloc[1:1025].tolist()

            # 진동 데이터: 2~513열 추출 (인덱스 1~513, 0-based) → 512개
            vibration_values = vibration_row.iloc[1:513].tolist()

            print(
                f"📊 전류 데이터: 행 {self.current_index}/{len(self.cached_current_df)} ({len(current_values)}개 값)")
            print(
                f"📊 진동 데이터: 행 {self.vibration_index}/{len(self.cached_vibration_df)} ({len(vibration_values)}개 값)")

            # API 호출용 데이터 형태로 변환
            return {
                "current": {
                    "signal_type": "cur",
                    "values": current_values
                },
                "vibration": {
                    "signal_type": "vib",
                    "values": vibration_values
                }
            }

        except Exception as e:
            print(f"❌ Welding Machine 데이터 시뮬레이션 실패: {e}")
            return None

    async def _load_dataframes(self):
        """DataFrame 캐싱을 위한 로드"""
        try:
            files = await self.list_data_files()

            if not files:
                print(f"⚠️ Welding Machine 데이터 파일이 없습니다.")
                return

            # 전류 및 진동 파일 찾기
            current_file = None
            vibration_file = None

            for file_path in files:
                if "current_augmented_test_with_labels.csv" in file_path:
                    current_file = file_path
                elif "vibration_augmented_test_with_labels.csv" in file_path:
                    vibration_file = file_path

            if not current_file or not vibration_file:
                print(f"⚠️ 전류 또는 진동 데이터 파일을 찾을 수 없습니다.")
                return

            # DataFrame 로드 및 캐싱
            self.cached_current_df = await self.read_csv_data(current_file)
            self.cached_vibration_df = await self.read_csv_data(vibration_file)

            # 인덱스 초기화
            self.current_index = 0
            self.vibration_index = 0

            print(
                f"✅ 데이터 캐싱 완료 - 전류: {len(self.cached_current_df)}행, 진동: {len(self.cached_vibration_df)}행")

        except Exception as e:
            print(f"❌ DataFrame 로드 실패: {e}")


# 글로벌 Azure Storage 서비스 인스턴스
azure_storage = AzureStorageService()
