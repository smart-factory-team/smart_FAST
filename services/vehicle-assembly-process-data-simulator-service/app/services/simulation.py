import asyncio
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from app.models.simulation import SimulationRequest
from app.services.azure_storage import AzureStorageService
from app.services.model_api import ModelAPIClient
from app.services.performance import PerformanceAnalyzer
import logging
import os

logger = logging.getLogger(__name__)

# 전역 시뮬레이션 저장소
simulations: Dict[str, dict] = {}

def load_ground_truth(file_path: str) -> Dict[str, str]:
    """Ground Truth 매핑 파일 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ground Truth 파일 로드 실패: {e}")
        return {}

class BatchSimulationProcessor:
    """배치 처리를 위한 시뮬레이션 프로세서"""

    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size

    async def download_batch_images(self, azure_service: AzureStorageService,
                                    filenames: List[str]) -> List[Tuple[str, Optional[bytes]]]:
        """배치로 이미지들 다운로드"""
        tasks = []
        for filename in filenames:
            task = self._download_single_image(azure_service, filename)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        batch_data = []
        for i, result in enumerate(results):
            filename = filenames[i]
            if isinstance(result, Exception):
                logger.error(f"이미지 다운로드 실패 {filename}: {result}")
                batch_data.append((filename, None))
            else:
                batch_data.append((filename, result))

        return batch_data

    async def _download_single_image(self, azure_service: AzureStorageService,
                                     filename: str) -> Optional[bytes]:
        """단일 이미지 다운로드"""
        return await azure_service.download_image(filename)

    async def predict_batch_images(self, api_client: ModelAPIClient,
                                   batch_data: List[Tuple[str, Optional[bytes]]]) -> List[Dict]:
        """배치로 이미지들 예측"""
        # 성공적으로 다운로드된 이미지들만 필터링
        valid_images = [(filename, image_data) for filename, image_data in batch_data if image_data is not None]
        failed_images = [(filename, image_data) for filename, image_data in batch_data if image_data is None]

        results = []

        # 실패한 이미지들 결과 추가
        for filename, _ in failed_images:
            results.append({
                "filename": filename,
                "predicted_label": None,
                "confidence": 0.0,
                "error": "이미지 다운로드 실패"
            })

        # 유효한 이미지들 병렬 예측
        if valid_images:
            batch_results = await self._predict_with_parallel(api_client, valid_images)
            results.extend(batch_results)
        return results

    async def _predict_with_batch_api(self, api_client: ModelAPIClient,
                                      valid_images: List[Tuple[str, bytes]]) -> List[Dict]:
        """배치 API를 사용한 예측"""
        try:
            # 모델 서버의 배치 API 호출
            filenames = [filename for filename, _ in valid_images]
            image_data_list = [image_data for _, image_data in valid_images]

            batch_response = await api_client.predict_batch(image_data_list, filenames)

            results = []
            for i, batch_item in enumerate(batch_response.data.results):
                if batch_item.success:
                    results.append({
                        "filename": batch_item.filename,
                        "predicted_label": batch_item.result.predicted_class,
                        "confidence": batch_item.result.confidence,
                        "error": None
                    })
                else:
                    results.append({
                        "filename": batch_item.filename,
                        "predicted_label": None,
                        "confidence": 0.0,
                        "error": batch_item.error
                    })

            return results

        except Exception as e:
            logger.error(f"배치 API 호출 실패: {e}")
            # 배치 API 실패시 병렬 처리로 폴백
            return await self._predict_with_parallel(api_client, valid_images)

    async def _predict_with_parallel(self, api_client: ModelAPIClient,
                                     valid_images: List[Tuple[str, bytes]]) -> List[Dict]:
        """병렬 처리를 사용한 예측"""
        tasks = []
        for filename, image_data in valid_images:
            task = api_client.predict(image_data, filename)
            tasks.append(task)

        predict_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, predict_result in enumerate(predict_results):
            filename = valid_images[i][0]

            if isinstance(predict_result, Exception):
                results.append({
                    "filename": filename,
                    "predicted_label": None,
                    "confidence": 0.0,
                    "error": str(predict_result)
                })
            else:
                predicted_label, confidence, error = predict_result
                results.append({
                    "filename": filename,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "error": error
                })

        return results


async def run_simulation(simulation_id: str, config: SimulationRequest):
    """배치 처리 개선된 시뮬레이션 실행"""
    try:
        # 상태 초기화
        simulations[simulation_id].update({
            "status": "running",
            "start_time": datetime.now(),
            "total_images": 0,
            "processed_images": 0,
            "results": [],
            "current_batch": 0,
            "total_batches": 0
        })

        # 서비스 초기화
        azure_service = AzureStorageService(config.azure_connection_string, config.container_name)
        await azure_service.connect()

        api_client = ModelAPIClient(config.model_api_url)
        analyzer = PerformanceAnalyzer()

        # 배치 프로세서 초기화 (batch_size는 config에서 가져오거나 기본값 사용)
        batch_size = getattr(config, 'batch_size', 10)
        batch_processor = BatchSimulationProcessor(batch_size=batch_size)

        # Ground Truth 로드
        ground_truth = load_ground_truth(config.test_mapping_path)
        if not ground_truth:
            raise Exception("Ground Truth 데이터를 로드할 수 없습니다")


        # 이미지 파일 목록 조회
        image_files = await azure_service.list_image_files(prefix=config.image_prefix)
        filtered_files = [f for f in image_files if os.path.basename(f) in ground_truth]

        if not filtered_files:
            raise Exception("처리할 이미지 파일이 없습니다")

        if config.limit:
            filtered_files = filtered_files[:config.limit]

        # 배치 정보 업데이트
        total_batches = (len(filtered_files) + batch_size - 1) // batch_size
        simulations[simulation_id].update({
            "total_images": len(filtered_files),
            "total_batches": total_batches
        })

        logger.info(f"시뮬레이션 {simulation_id}: {len(filtered_files)}개 이미지를 {total_batches}개 배치로 처리 시작")

        # 배치 단위로 처리
        all_results = []

        for batch_idx in range(0, len(filtered_files), batch_size):
            batch_files = filtered_files[batch_idx:batch_idx + batch_size]
            current_batch_num = (batch_idx // batch_size) + 1

            logger.info(f"배치 {current_batch_num}/{total_batches} 처리 중 ({len(batch_files)}개 이미지)")

            try:
                # 1. 배치 이미지 다운로드
                batch_images = await batch_processor.download_batch_images(azure_service, batch_files)

                # 2. 배치 예측 수행
                batch_predictions = await batch_processor.predict_batch_images(api_client, batch_images)

                # API 결과 상세 로깅
                logger.info(f"📊 API 예측 결과 수신: {len(batch_predictions)}개")
                for i, prediction in enumerate(batch_predictions):
                    logger.info(f"🔍 예측 결과 {i+1}:")
                    logger.info(f"   파일명: {prediction['filename']}")
                    logger.info(f"   예측값: {prediction['predicted_label']}")
                    logger.info(f"   신뢰도: {prediction['confidence']}")
                    logger.info(f"   오류: {prediction.get('error', '없음')}")

                # 3. 결과 처리 및 Ground Truth 비교
                logger.info(f"🔗 Ground Truth 매칭 시작")
                for prediction in batch_predictions:
                    filename = prediction["filename"]
                    predicted_label = prediction["predicted_label"]

                    # 파일명만 추출해서 Ground Truth와 매칭
                    filename_only = os.path.basename(filename)
                    gt_label = ground_truth.get(filename_only, "unknown")

                    # 매칭 결과 로깅
                    logger.info(f"📋 매칭: {filename_only}")
                    logger.info(f"   예측: {predicted_label}")
                    logger.info(f"   정답: {gt_label}")
                    logger.info(f"   일치: {'✅' if predicted_label == gt_label else '❌'}")

                    result = {
                        "filename": filename,
                        "predicted_label": predicted_label,
                        "confidence": prediction["confidence"],
                        "ground_truth": gt_label,
                        "is_correct": predicted_label == gt_label if predicted_label else False,
                        "error": prediction["error"]
                    }

                    all_results.append(result)

                # 진행률 업데이트
                simulations[simulation_id].update({
                    "processed_images": len(all_results),
                    "current_batch": current_batch_num,
                    "results": all_results
                })

                # 현재까지의 정확도 계산
                if all_results:
                    current_metrics = analyzer.calculate_metrics(all_results)
                    simulations[simulation_id]["accuracy"] = current_metrics["accuracy"]

                    # 배치 요약 로그
                    batch_correct = sum(1 for r in all_results[-len(batch_predictions):] if r['is_correct'])
                    logger.info(f"🎯 배치 {current_batch_num} 완료:")
                    logger.info(f"   처리된 파일: {len(batch_predictions)}개")
                    logger.info(f"   정확한 예측: {batch_correct}개")
                    logger.info(f"   배치 정확도: {batch_correct/len(batch_predictions)*100:.1f}%")
                    logger.info(f"   전체 진행률: {len(all_results)}/{len(filtered_files)} ({len(all_results)/len(filtered_files)*100:.1f}%)")
                    logger.info(f"   누적 정확도: {current_metrics['accuracy']:.4f}")

                # 만약 API 호출 자체가 실패한다면
                if not batch_predictions:
                    logger.warning("⚠️  API에서 결과를 받지 못했습니다!")
                elif len(batch_predictions) != len(batch_files):
                    logger.warning(f"⚠️  요청 파일 수({len(batch_files)})와 응답 수({len(batch_predictions)})가 다릅니다!")

                # 예측 실패한 파일들 별도 확인
                failed_predictions = [p for p in batch_predictions if p['predicted_label'] is None]
                if failed_predictions:
                    logger.warning(f"❌ 예측 실패한 파일들 ({len(failed_predictions)}개):")
                    for failed in failed_predictions:
                        logger.warning(f"   - {failed['filename']}: {failed.get('error', '알 수 없는 오류')}")

                # Ground Truth 매칭 실패한 파일들
                unknown_gt = [r for r in all_results[-len(batch_predictions):] if r['ground_truth'] == 'unknown']
                if unknown_gt:
                    logger.warning(f"🔍 Ground Truth 매칭 실패 ({len(unknown_gt)}개):")
                    for unknown in unknown_gt:
                        filename_only = os.path.basename(unknown['filename'])
                        logger.warning(f"   - {filename_only}: Ground Truth에 없음")


            except Exception as e:
                logger.error(f"배치 {current_batch_num} 처리 중 오류: {e}")
                # 배치 실패시에도 계속 진행
                continue

        # 최종 결과 계산
        if all_results:
            final_metrics = analyzer.calculate_metrics(all_results)
            simulations[simulation_id].update({
                "status": "completed",
                "end_time": datetime.now(),
                "final_metrics": final_metrics,
                "results": all_results
            })

            processing_time = (datetime.now() - simulations[simulation_id]["start_time"]).total_seconds()
            logger.info(f"시뮬레이션 {simulation_id} 완료 - "
                        f"총 {len(all_results)}개 이미지 처리, "
                        f"최종 정확도: {final_metrics['accuracy']:.4f}, "
                        f"처리 시간: {processing_time:.2f}초")
            # 디버깅: Ground Truth 매칭 확인
            for result in batch_predictions:
                filename_only = os.path.basename(result["filename"])
                gt_label = ground_truth.get(filename_only, "NOT_FOUND")
                logger.debug(f"매칭 확인: {filename_only} → {gt_label}")
        else:
            raise Exception("처리된 결과가 없습니다")

        await azure_service.disconnect()

    except Exception as e:
        simulations[simulation_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now()
        })
        logger.error(f"시뮬레이션 {simulation_id} 실패: {e}")

        # Azure 연결 정리
        try:
            await azure_service.disconnect()
        except:
            pass


# 배치 크기 설정을 위한 SimulationRequest 확장이 필요한 경우
# models/simulation.py에 batch_size: int = 10 필드 추가 권장