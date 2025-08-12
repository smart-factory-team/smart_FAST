import pandas as pd
from typing import List, Dict, Any, Optional
from azure.storage.blob.aio import BlobServiceClient
from app.config.settings import settings
import io
import logging

logger = logging.getLogger(__name__)

class AzureStorageService:
    def __init__(self):
        self.connection_string = settings.azure_connection_string
        self.container_name = settings.azure_container_name
        self.blob_service_client = None
        if self.connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

        # 순차 처리를 위한 인덱스 관리
        self.current_index = 0
        self.cached_df = None

    async def list_data_files(self) -> List[str]:
        """데이터 파일 목록 조회"""
        if not self.blob_service_client:
            raise ValueError("Azure connection string이 설정되지 않았습니다.")
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = []

            # painting-process-equipment 폴더만 검색
            prefix = f"{settings.painting_data_folder}/"

            async for blob in container_client.list_blobs(name_starts_with=prefix):
                if blob.name.endswith('.csv'):
                    blob_list.append(blob.name)

            return sorted(blob_list)
        except Exception as e:
            logger.error(f"❌ Painting 데이터 파일 목록 조회 실패: {e}")
            return []

    async def read_csv_data(self, blob_name: str) -> Optional[pd.DataFrame]:
        """CSV 파일 읽기"""
        if not self.blob_service_client:
            raise ValueError("Azure connection string이 설정되지 않았습니다.")

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            blob_data = await blob_client.download_blob()
            content = await blob_data.readall()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))

            logger.info(f"📁 파일 읽기 성공: {blob_name} ({len(df)} rows)")
            return df

        except Exception as e:
            logger.error(f"❌ 파일 읽기 실패 ({blob_name}): {e}")
            return None

    async def simulate_real_time_data(self) -> Optional[Dict[str, Any]]:
        """Painting Process Equipment 실시간 데이터 시뮬레이션 - 순차 처리"""
        try:
            if self.cached_df is None:
                await self._load_dataframe()

            if self.cached_df is None:
                return None

            # 순차적으로 행 선택
            data_row = self.cached_df.iloc[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.cached_df)

            # 필요한 모든 열이 존재하는지 확인
            required_columns = ['machineId', 'timeStamp', 'Thick', 'PT_jo_V_1', 'PT_jo_A_Main_1', 'PT_jo_TP']
            missing_columns = [col for col in required_columns if col not in data_row.index]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # API 호출용 데이터 형태로 변환
            simulated_data = {
                "machineId": f"PAINT-{data_row.get('machineId')}",
                "timeStamp": data_row.get("timeStamp"),
                "thick": float(data_row.get("Thick")),
                "voltage": float(data_row.get("PT_jo_V_1")),
                "current": float(data_row.get("PT_jo_A_Main_1")),
                "temper": float(data_row.get("PT_jo_TP")),
                "issue": "",  # 모델 서비스에서 채워짐
                "isSolved": False # 기본값
            }
            
            logger.info(f"📊 Painting 데이터: 행 {self.current_index}/{len(self.cached_df)}")
            return simulated_data

        except Exception as e:
            logger.error(f"❌ Painting 데이터 시뮬레이션 실패: {e}")
            return None

    async def _load_dataframe(self):
        """DataFrame 캐싱을 위한 로드"""
        try:
            files = await self.list_data_files()
            if not files:
                logger.warning("⚠️ Painting 데이터 파일이 없습니다.")
                return

            # 첫 번째 CSV 파일을 사용
            target_file = files[0]
            logger.info(f"✅ 시뮬레이션에 사용할 파일: {target_file}. 발견된 파일들 중 첫번째 파일을 사용합니다.")
            
            self.cached_df = await self.read_csv_data(target_file)
            self.current_index = 0

            if self.cached_df is not None:
                logger.info(f"✅ 데이터 캐싱 완료: {target_file} ({len(self.cached_df)} rows)")

        except Exception as e:
            logger.error(f"❌ DataFrame 로드 실패: {e}")


# 글로벌 Azure Storage 서비스 인스턴스
azure_storage = AzureStorageService()