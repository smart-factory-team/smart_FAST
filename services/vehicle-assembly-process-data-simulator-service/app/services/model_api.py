import aiohttp
import logging
from typing import Tuple, Optional, List, Dict

logger = logging.getLogger(__name__)

class ModelAPIClient:
    def __init__(self, api_url: str, timeout: int = 30):
        self.api_url = api_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def predict(self, image_data: bytes, filename: str) -> Tuple[Optional[str], float, Optional[str]]:
        """모델 예측 요청"""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                form_data = aiohttp.FormData()
                form_data.add_field('file', image_data, filename=filename, content_type='image/jpeg')

                async with session.post(f"{self.api_url}/predict/file", data=form_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        data = result.get('data', {})
                        return data.get('predicted_category_name'), data.get('confidence', 0.0), None
                    else:
                        return None, 0.0, f"API 오류: {response.status}"
        except Exception as e:
            logger.error(f"모델 예측 실패 ({filename}): {e}")
            return None, 0.0, str(e)

    async def predict_batch(self, image_data_list: List[bytes], filenames: List[str]) -> Dict:
        """배치 이미지 예측 (모델 서버의 배치 API 호출)"""
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                form_data = aiohttp.FormData()

                # 여러 파일을 FormData에 추가
                for i, (image_data, filename) in enumerate(zip(image_data_list, filenames)):
                    form_data.add_field('files', image_data, filename=filename, content_type='image/jpeg')

                async with session.post(f"{self.api_url}/predict_batch", data=form_data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"배치 API 응답 오류: {response.status}")

        except Exception as e:
            raise Exception(f"배치 API 호출 오류: {e}")
