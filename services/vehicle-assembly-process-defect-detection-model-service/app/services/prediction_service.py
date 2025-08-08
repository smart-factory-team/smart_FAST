"""
예측 서비스 모듈

AI 모델을 사용한 이미지 불량 예측 기능을 제공합니다.
- 단일 이미지 예측
- 파일 업로드 예측
- 배치 예측
- 이미지 전처리 및 검증
"""

import asyncio
import io
import time
from typing import Dict, Any, Optional, List, Tuple, Union

import logging
from datetime import datetime

import aiohttp
from fastapi import UploadFile
from PIL import Image

from app.core.config import settings
from app.core.exceptions import (
    InvalidImageException,
    ImageSizeException,
    UnsupportedImageFormatException
)
from app.schemas.predict import (
    PredictRequest,
    PredictResponse,
    FilePredictResponse,
    BatchPredictResponse,
    PredictionResult,
    FilePredictionResult,
    BatchPredictionData,
    BatchPredictionItem,
    BatchSummary,
    ClassProbability
)
from app.schemas.common import ProcessingTime, FileInfo


class PredictionService:
    """
    예측 서비스 클래스

    AI 모델을 사용하여 이미지의 불량을 탐지하는 서비스입니다.
    다양한 입력 형태(JSON, 파일, 배치)를 지원합니다.
    """

    def __init__(self, model_manager):
        """
        예측 서비스 초기화

        Args:
            model_manager: DefectDetectionModelManager 인스턴스
        """
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

        # 지원하는 이미지 형식
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # 이미지 크기 제한
        self.max_image_size = settings.MAX_IMAGE_SIZE
        self.max_dimension = 4096  # 최대 가로/세로 크기

    async def predict_from_data(self, request: PredictRequest) -> PredictResponse:
        """
        JSON 데이터로부터 예측 수행

        Args:
            request: 예측 요청 데이터

        Returns:
            PredictResponse: 예측 결과
        """
        start_time = time.time()

        try:
            # 이미지 로드
            image = await self._load_image_from_request(request.image)

            # 이미지 검증
            self._validate_image(image)

            # 예측 수행
            prediction_result = await self.model_manager.predict(image)

            # 응답 데이터 구성
            result_data = PredictionResult(
                predicted_category_id=prediction_result["predicted_category_id"],
                predicted_category_name=prediction_result["predicted_category_name"],
                predicted_label=prediction_result["predicted_label"],
                confidence=prediction_result["confidence"],
                is_defective=prediction_result["is_defective"],
                class_probabilities=[
                    ClassProbability(**prob)
                    for prob in prediction_result.get("class_probabilities", [])
                ] if settings.RETURN_CLASS_PROBABILITIES else None,
                processing_time=ProcessingTime(**prediction_result["processing_time"]),
                model_version=prediction_result.get("model_version")
            )

            return PredictResponse(
                success=True,
                message="예측이 성공적으로 완료되었습니다",
                data=result_data
            )

        except Exception as e:
            self.logger.error(f"JSON 데이터 예측 중 오류: {e}")
            raise

    async def predict_from_file(self, file: UploadFile) -> FilePredictResponse:
        """
        업로드된 파일로부터 예측 수행

        Args:
            file: 업로드된 이미지 파일

        Returns:
            FilePredictResponse: 파일 예측 결과
        """
        start_time = time.time()

        try:
            self.logger.info("파일 데이터 읽는 중...")
            # 파일에서 이미지 로드
            image = await self._load_image_from_file(file)

            self.logger.info("이미지 검증 시작...")
            # 이미지 검증
            self._validate_image(image)

            self.logger.info("파일 정보 수집 중...")
            # 파일 정보 수집
            file_info = await self._get_file_info(file, image)

            self.logger.info("예측 시작...")
            # 예측 수행
            prediction_result = await self.model_manager.predict(image)

            # 응답 데이터 구성
            result_data = FilePredictionResult(
                predicted_category_id=prediction_result["predicted_category_id"],
                predicted_category_name=prediction_result["predicted_category_name"],
                predicted_label=prediction_result["predicted_label"],
                confidence=prediction_result["confidence"],
                is_defective=prediction_result["is_defective"],
                class_probabilities=[
                    ClassProbability(**prob)
                    for prob in prediction_result.get("class_probabilities", [])
                ] if settings.RETURN_CLASS_PROBABILITIES else None,
                processing_time=ProcessingTime(**prediction_result["processing_time"]),
                model_version=prediction_result.get("model_version"),
                file_info=file_info
            )

            return FilePredictResponse(
                success=True,
                message="파일 예측이 성공적으로 완료되었습니다",
                data=result_data
            )

        except Exception as e:
            self.logger.error(f"파일 예측 중 오류: {e}")
            raise

    async def predict_batch(self, files: List[UploadFile]) -> BatchPredictResponse:
        """
        배치 파일 예측 수행

        Args:
            files: 업로드된 이미지 파일들

        Returns:
            BatchPredictResponse: 배치 예측 결과
        """
        start_time = time.time()

        try:
            # 배치 예측 수행
            batch_results = []
            successful_count = 0
            failed_count = 0
            defective_count = 0
            normal_count = 0
            total_confidence = 0.0
            confidence_count = 0

            # 병렬 처리를 위한 태스크들
            tasks = []
            for file in files:
                task = self._predict_single_file(file)
                tasks.append(task)

            # 모든 예측 병렬 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 처리
            for i, result in enumerate(results):
                filename = files[i].filename or f"file_{i}"

                if isinstance(result, Exception):
                    # 예측 실패
                    batch_results.append(BatchPredictionItem(
                        filename=filename,
                        success=False,
                        result=None,
                        error=str(result)
                    ))
                    failed_count += 1
                else:
                    # 예측 성공
                    batch_results.append(BatchPredictionItem(
                        filename=filename,
                        success=True,
                        result=result,
                        error=None
                    ))
                    successful_count += 1

                    # 통계 업데이트
                    if result.is_defective:
                        defective_count += 1
                    else:
                        normal_count += 1

                    total_confidence += result.confidence
                    confidence_count += 1

            # 평균 신뢰도 계산
            average_confidence = (
                total_confidence / confidence_count
                if confidence_count > 0 else None
            )

            total_processing_time = time.time() - start_time

            # 요약 정보
            summary = BatchSummary(
                total_count=len(files),
                successful_count=successful_count,
                failed_count=failed_count,
                defective_count=defective_count,
                normal_count=normal_count,
                average_confidence=average_confidence,
                total_processing_time=total_processing_time
            )

            # 배치 데이터 구성
            batch_data = BatchPredictionData(
                results=batch_results,
                summary=summary
            )

            return BatchPredictResponse(
                success=True,
                message=f"배치 예측 완료: {successful_count}개 성공, {failed_count}개 실패",
                data=batch_data
            )

        except Exception as e:
            self.logger.error(f"배치 예측 중 오류: {e}")
            raise

    async def _predict_single_file(self, file: UploadFile) -> FilePredictionResult:
        """
        단일 파일 예측 (배치 처리용)

        Args:
            file: 업로드된 파일

        Returns:
            FilePredictionResult: 예측 결과
        """
        try:
            # 파일에서 이미지 로드
            image = await self._load_image_from_file(file)

            # 이미지 검증
            self._validate_image(image)

            # 파일 정보 수집
            file_info = await self._get_file_info(file, image)

            # 예측 수행
            prediction_result = await self.model_manager.predict(image)

            # 결과 구성
            return FilePredictionResult(
                predicted_category_id=prediction_result["predicted_category_id"],
                predicted_category_name=prediction_result["predicted_category_name"],
                predicted_label=prediction_result["predicted_label"],
                confidence=prediction_result["confidence"],
                is_defective=prediction_result["is_defective"],
                class_probabilities=[
                    ClassProbability(**prob)
                    for prob in prediction_result.get("class_probabilities", [])
                ] if settings.RETURN_CLASS_PROBABILITIES else None,
                processing_time=ProcessingTime(**prediction_result["processing_time"]),
                model_version=prediction_result.get("model_version"),
                file_info=file_info
            )

        except Exception as e:
            self.logger.error(f"단일 파일 예측 중 오류: {e}")
            raise

    async def _load_image_from_request(self, image_data) -> Image.Image:
        """
        요청 데이터에서 이미지 로드 (URL만 지원)

        Args:
            image_data: 이미지 데이터 (URL)

        Returns:
            Image.Image: PIL 이미지 객체
        """
        if image_data.url:
            return await self._load_image_from_url(str(image_data.url))
        else:
            raise InvalidImageException("이미지 URL이 제공되지 않았습니다")

    async def _load_image_from_url(self, url: str) -> Image.Image:
        """
        URL에서 이미지 로드

        Args:
            url: 이미지 URL

        Returns:
            Image.Image: PIL 이미지 객체
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)  # 10초 타임아웃

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise InvalidImageException(
                            f"이미지 다운로드 실패: HTTP {response.status}"
                        )

                    # Content-Length 확인
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > self.max_image_size:
                        raise ImageSizeException(int(content_length), self.max_image_size)

                    # 이미지 데이터 읽기
                    image_data = await response.read()

                    # 크기 재확인
                    if len(image_data) > self.max_image_size:
                        raise ImageSizeException(len(image_data), self.max_image_size)

                    # PIL 이미지로 변환
                    image = Image.open(io.BytesIO(image_data))

                    return image

        except aiohttp.ClientError as e:
            self.logger.error(f"URL 이미지 로드 실패: {e}")
            raise InvalidImageException(f"URL 이미지 로드 실패: {str(e)}") from e
        except Exception as e:
            self.logger.error(f"URL 이미지 처리 실패: {e}")
            raise InvalidImageException(f"URL 이미지 처리 실패: {str(e)}") from e

    async def _load_image_from_file(self, file: UploadFile) -> Image.Image:
        """
        업로드된 파일에서 이미지 로드

        Args:
            file: 업로드된 파일

        Returns:
            Image.Image: PIL 이미지 객체
        """
        try:
            # 파일 내용 읽기
            contents = await file.read()

            # 크기 확인
            if len(contents) > self.max_image_size:
                raise ImageSizeException(len(contents), self.max_image_size)

            # PIL 이미지로 변환
            image = Image.open(io.BytesIO(contents))

            # 파일 포인터 리셋
            await file.seek(0)

            return image

        except Exception as e:
            self.logger.error(f"파일 이미지 로드 실패: {e}")
            raise InvalidImageException(f"파일 이미지 로드 실패: {str(e)}") from e

    def _validate_image(self, image: Image.Image):
        """
        이미지 유효성 검증

        Args:
            image: PIL 이미지 객체

        Raises:
            InvalidImageException: 이미지가 유효하지 않은 경우
            ImageSizeException: 이미지 크기가 제한을 초과하는 경우
        """
        # 이미지 형식 확인
        if image.format and image.format.upper() not in ['JPEG', 'PNG', 'BMP', 'TIFF']:
            raise UnsupportedImageFormatException(
                image.format,
                ['JPEG', 'PNG', 'BMP', 'TIFF']
            )

        # 이미지 크기 확인
        width, height = image.size
        if width > self.max_dimension or height > self.max_dimension:
            raise ImageSizeException(
                f"{width}x{height}",
                f"{self.max_dimension}x{self.max_dimension}"
            )

        # 최소 크기 확인
        if width < 32 or height < 32:
            raise InvalidImageException("이미지가 너무 작습니다 (최소 32x32 픽셀)")

        # 이미지 모드 확인 및 변환
        if image.mode not in ['RGB', 'RGBA', 'L']:
            try:
                image = image.convert('RGB')
            except Exception:
                raise InvalidImageException("지원하지 않는 이미지 색상 모드입니다") from None

    async def _get_file_info(self, file: UploadFile, image: Image.Image) -> FileInfo:
        """
        파일 정보 수집

        Args:
            file: 업로드된 파일
            image: PIL 이미지 객체

        Returns:
            FileInfo: 파일 정보
        """
        width, height = image.size

        return FileInfo(
            filename=file.filename,
            content_type=file.content_type,
            size=file.size,
            width=width,
            height=height
        )

    def get_service_status(self) -> Dict[str, Any]:
        """
        서비스 상태 정보 반환

        Returns:
            Dict: 서비스 상태 정보
        """
        return {
            "model_loaded": self.model_manager.is_loaded,
            "model_status": self.model_manager.status.value,
            "supported_formats": list(self.supported_formats),
            "max_image_size_mb": self.max_image_size / (1024 * 1024),
            "max_dimension": self.max_dimension,
            "batch_size_limit": settings.BATCH_SIZE_LIMIT,
            "prediction_timeout": settings.PREDICTION_TIMEOUT
        }