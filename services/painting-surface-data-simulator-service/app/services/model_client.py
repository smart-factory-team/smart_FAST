import httpx
import asyncio
from typing import Dict, Any, Optional, List
from app.config.settings import settings
from datetime import datetime


class PaintingSurfaceModelClient:
    """도장 표면 결함탐지 전용 모델 클라이언트"""
    
    def __init__(self):
        self.timeout = httpx.Timeout(settings.http_timeout)
        self.max_retries = settings.max_retries
        self.service_name = "painting-surface"
        self.service_url = settings.model_service_url

    async def predict_painting_surface_data(self, image_files: List[str]) -> Optional[Dict[str, Any]]:
        """도장 표면 이미지 데이터 예측 요청"""
        if not self.service_url:
            print(f"❌ Painting Surface 서비스 URL을 찾을 수 없습니다.")
            return None

        # 파일 업로드 방식 엔드포인트 사용
        predict_url = f"{self.service_url}/predict/file"

        results = {}

        # 각 이미지 파일에 대해 예측 요청
        print(f"🎨 도장 표면 이미지 예측 요청...")
        image_results = []
        
        for image_file in image_files:
            try:
                # Azure Storage에서 이미지 다운로드
                print(f"📥 이미지 다운로드 중: {image_file}")
                image_data = await self._download_image_from_azure(image_file)
                
                if not image_data:
                    print(f"⚠️ 이미지 다운로드 실패: {image_file}")
                    continue
                
                # 파일 업로드 방식으로 예측 요청
                result = await self._predict_with_file_upload(predict_url, image_data, image_file, 0.5)
                if result:
                    # 상세한 예측 결과 로깅
                    self._log_detailed_prediction_result(image_file, result)
                    image_results.append(result)
                
            except Exception as e:
                print(f"❌ 이미지 {image_file} 예측 실패: {e}")
                continue

        if not image_results:
            print("❌ 모든 이미지 예측이 실패했습니다.")
            return None

        # 결과 조합
        combined_result = self._combine_painting_results(image_results)
        results["images"] = image_results
        results["combined"] = combined_result

        return results

    def _combine_painting_results(self, image_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """도장 표면 이미지 결과 조합"""
        if not image_results:
            return {
                "status": "error",
                "message": "예측 결과를 받을 수 없습니다."
            }

        # 결함이 하나라도 탐지되면 anomaly
        defect_count = sum(1 for result in image_results if result.get('predictions', []))
        total_count = len(image_results)

        if defect_count > 0:
            final_status = "anomaly"
        else:
            final_status = "normal"

        return {
            "status": final_status,
            "defect_count": defect_count,
            "total_count": total_count,
            "defect_ratio": defect_count / total_count if total_count > 0 else 0,
            "combined_logic": f"총 {total_count}개 이미지 중 {defect_count}개에서 결함 탐지 → 최종: {final_status}"
        }

    async def health_check(self) -> bool:
        """도장 표면 결함탐지 서비스 헬스 체크"""
        if not self.service_url:
            return False

        health_url = f"{self.service_url}/health"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(health_url)
                return response.status_code == 200
        except:
            return False

    async def _download_image_from_azure(self, image_path: str) -> Optional[bytes]:
        """Azure Storage에서 이미지 다운로드"""
        try:
            # azure_storage 서비스에서 이미지 데이터 읽기
            from app.services.azure_storage import azure_storage
            image_data = await azure_storage.read_image_data(image_path)
            return image_data
        except Exception as e:
            print(f"❌ Azure Storage에서 이미지 다운로드 실패 ({image_path}): {e}")
            return None

    async def _predict_with_file_upload(self, predict_url: str, image_data: bytes, filename: str, confidence_threshold: float) -> Optional[Dict[str, Any]]:
        """파일 업로드 방식으로 예측 요청"""
        for attempt in range(self.max_retries):
            try:
                # multipart/form-data로 파일 업로드
                files = {
                    'image': (filename, image_data, 'image/jpeg')
                }
                data = {
                    'confidence_threshold': str(confidence_threshold)
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(predict_url, files=files, data=data)

                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ {filename} 예측 성공 (시도 {attempt + 1})")
                        return result
                    else:
                        print(f"⚠️ {filename} HTTP {response.status_code} (시도 {attempt + 1})")

            except httpx.TimeoutException:
                print(f"⏰ {filename} 타임아웃 (시도 {attempt + 1})")
            except httpx.ConnectError:
                print(f"🔌 {filename} 연결 실패 (시도 {attempt + 1})")
            except Exception as e:
                print(f"❌ {filename} 예측 오류 (시도 {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # 지수 백오프

        print(f"❌ {filename} 최대 재시도 횟수 초과")
        return None

    def _log_detailed_prediction_result(self, image_file: str, result: Dict[str, Any]):
        """상세한 예측 결과 로깅"""
        print(f"\n🔍 {image_file} 상세 예측 결과:")
        print(f"   📊 이미지 크기: {result.get('image_shape', 'N/A')}")
        print(f"   🎯 신뢰도 임계값: {result.get('confidence_threshold', 'N/A')}")
        
        predictions = result.get('predictions', [])
        if predictions:
            print(f"   ⚠️  결함 탐지됨: {len(predictions)}개")
            for i, pred in enumerate(predictions, 1):
                print(f"      결함 {i}:")
                print(f"        🏷️  종류: {pred.get('class_name', 'N/A')}")
                print(f"        📍 위치: {pred.get('bbox', 'N/A')}")
                print(f"        📏 크기: {pred.get('area', 'N/A')} 픽셀²")
                print(f"        🎯 신뢰도: {pred.get('confidence', 'N/A'):.3f}")
        else:
            print(f"   ✅ 결함 없음 - 정상 상태")
        
        print(f"   🕒 예측 시간: {result.get('timestamp', 'N/A')}")
        print(f"   🤖 모델 소스: {result.get('model_source', 'N/A')}")
        print("-" * 60)


# 글로벌 도장 표면 모델 클라이언트 인스턴스
painting_surface_model_client = PaintingSurfaceModelClient()
