"""
모델 서비스 모듈

AI 모델의 정보 조회, 관리, 모니터링 기능을 제공합니다.
- 모델 정보 조회
- 클래스 정보 조회
- 성능 지표 조회
- 모델 재로딩
- 사용 통계 관리
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

from datetime import datetime, timedelta

from app.schemas.model import (
    ModelInfo,
    ModelClassesInfo,
    PerformanceMetrics,
    ModelReloadResult
)
from app.schemas.common import ModelStatus
from app.core.config import settings


class ModelService:
    """
    모델 서비스 클래스

    AI 모델의 메타데이터, 성능 지표, 사용 통계 등을 관리하는 서비스입니다.
    """

    def __init__(self, model_manager):
        """
        모델 서비스 초기화

        Args:
            model_manager: DefectDetectionModelManager 인스턴스
        """
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

        # 성능 지표 캐시 (실제 환경에서는 데이터베이스나 Redis 사용)
        self._performance_cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(hours=1)  # 1시간 캐시

    async def get_model_info(self) -> ModelInfo:
        """
        모델 상세 정보 조회

        Returns:
            ModelInfo: 모델 상세 정보
        """
        try:
            self.logger.info("모델 정보 조회 시작")

            if not self.model_manager:
                raise RuntimeError("모델 매니저가 초기화되지 않았습니다")

            # 모델 매니저에서 정보 수집
            model_info = await self.model_manager.get_model_info()

            self.logger.info("모델 정보 조회 완료")
            return model_info

        except Exception as e:
            self.logger.error(f"모델 정보 조회 중 오류: {e}")
            raise

    async def get_model_classes(self) -> ModelClassesInfo:
        """
        모델이 지원하는 클래스 정보 조회

        Returns:
            ModelClassesInfo: 클래스 정보
        """
        try:
            self.logger.info("모델 클래스 정보 조회 시작")

            if not self.model_manager:
                raise RuntimeError("모델 매니저가 초기화되지 않았습니다")

            # 모델 매니저에서 클래스 정보 수집
            classes_info = await self.model_manager.get_model_classes()

            self.logger.info("모델 클래스 정보 조회 완료")
            return classes_info

        except Exception as e:
            self.logger.error(f"모델 클래스 조회 중 오류: {e}")
            raise

    async def get_model_performance(self) -> PerformanceMetrics:
        """
        모델 성능 지표 조회

        캐시된 성능 지표를 반환하거나, 캐시가 없으면 새로 계산합니다.

        Returns:
            PerformanceMetrics: 성능 지표
        """
        try:
            self.logger.info("모델 성능 지표 조회 시작")

            # 캐시 확인
            if self._is_performance_cache_valid():
                self.logger.info("캐시된 성능 지표 반환")
                return PerformanceMetrics(**self._performance_cache)

            if not self.model_manager:
                raise RuntimeError("모델 매니저가 초기화되지 않았습니다")

            # 모델 매니저에서 성능 지표 수집
            performance_metrics = await self.model_manager.get_model_performance()

            # 실시간 성능 지표 추가 (실제 예측 통계 기반)
            if self.model_manager.statistics:
                # 실시간 통계 반영
                real_time_metrics = self._calculate_realtime_metrics()
                if real_time_metrics:
                    # 기존 성능 지표와 실시간 지표 결합
                    performance_metrics = self._merge_performance_metrics(
                        performance_metrics,
                        real_time_metrics
                    )

            # 캐시 업데이트
            self._update_performance_cache(performance_metrics)

            self.logger.info("모델 성능 지표 조회 완료")
            return performance_metrics

        except Exception as e:
            self.logger.error(f"모델 성능 조회 중 오류: {e}")
            raise

    async def reload_model(self) -> ModelReloadResult:
        """
        모델 재로딩 수행

        Returns:
            ModelReloadResult: 재로딩 결과
        """
        try:
            self.logger.info("모델 재로딩 시작")

            if not self.model_manager:
                raise RuntimeError("모델 매니저가 초기화되지 않았습니다")

            # 재로딩 전 상태 저장
            old_status = self.model_manager.status
            old_version = self.model_manager.model_version

            # 모델 재로딩 수행
            reload_result = await self.model_manager.reload_model()

            # 성능 캐시 초기화
            self._clear_performance_cache()

            # 결과 구성
            if reload_result["success"]:
                result = ModelReloadResult(
                    success=True,
                    old_version=reload_result["old_version"],
                    new_version=reload_result["new_version"],
                    reload_time_seconds=reload_result["reload_time_seconds"],
                    changes=reload_result.get("changes", [])
                )
                self.logger.info(f"모델 재로딩 성공: {old_version} -> {result.new_version}")
            else:
                result = ModelReloadResult(
                    success=False,
                    old_version=reload_result["old_version"],
                    new_version=reload_result["new_version"],
                    reload_time_seconds=reload_result["reload_time_seconds"],
                    changes=reload_result.get("changes", [])
                )
                self.logger.error("모델 재로딩 실패")

            return result

        except Exception as e:
            self.logger.error(f"모델 재로딩 중 오류: {e}")
            raise

    def _is_performance_cache_valid(self) -> bool:
        """
        성능 지표 캐시 유효성 확인

        Returns:
            bool: 캐시 유효 여부
        """
        if not self._performance_cache or not self._cache_timestamp:
            return False

        return datetime.now() - self._cache_timestamp < self._cache_duration

    def _update_performance_cache(self, performance_metrics: PerformanceMetrics):
        """
        성능 지표 캐시 업데이트

        Args:
            performance_metrics: 업데이트할 성능 지표
        """
        self._performance_cache = performance_metrics.dict()
        self._cache_timestamp = datetime.now()

    def _clear_performance_cache(self):
        """성능 지표 캐시 초기화"""
        self._performance_cache = {}
        self._cache_timestamp = None

    def _calculate_realtime_metrics(self) -> Optional[Dict[str, Any]]:
        """
        실시간 성능 지표 계산

        실제 예측 결과를 바탕으로 실시간 성능 지표를 계산합니다.

        Returns:
            Optional[Dict]: 실시간 성능 지표
        """
        try:
            stats = self.model_manager.statistics

            if stats.total_predictions == 0:
                return None

            # 에러율 기반 성능 추정
            success_rate = 1.0 - stats.error_rate

            # 클래스별 예측 분포 분석
            class_distribution = stats.class_prediction_counts
            total_class_predictions = sum(class_distribution.values())

            # 정상/불량 비율 계산
            normal_predictions = class_distribution.get("정상", 0)
            defect_predictions = total_class_predictions - normal_predictions

            # 간단한 성능 지표 추정
            estimated_metrics = {
                "estimated_accuracy": success_rate,
                "total_predictions": stats.total_predictions,
                "success_rate": success_rate,
                "error_rate": stats.error_rate,
                "average_processing_time": stats.average_processing_time,
                "normal_detection_rate": normal_predictions / total_class_predictions if total_class_predictions > 0 else 0,
                "defect_detection_rate": defect_predictions / total_class_predictions if total_class_predictions > 0 else 0,
                "class_distribution": class_distribution
            }

            return estimated_metrics

        except Exception as e:
            self.logger.warning(f"실시간 성능 지표 계산 실패: {e}")
            return None

    def _merge_performance_metrics(
        self,
        base_metrics: PerformanceMetrics,
        realtime_metrics: Dict[str, Any]
    ) -> PerformanceMetrics:
        """
        기본 성능 지표와 실시간 지표 병합

        Args:
            base_metrics: 기본 성능 지표
            realtime_metrics: 실시간 성능 지표

        Returns:
            PerformanceMetrics: 병합된 성능 지표
        """
        try:
            # 기본 지표를 딕셔너리로 변환
            merged = base_metrics.dict()

            # 실시간 지표로 업데이트 (가중 평균 사용)
            if realtime_metrics.get("estimated_accuracy") and merged.get("accuracy"):
                # 기존 정확도와 실시간 추정 정확도의 가중 평균
                weight = min(realtime_metrics["total_predictions"] / 1000, 0.3)  # 최대 30% 가중치
                merged["accuracy"] = (
                    merged["accuracy"] * (1 - weight) +
                    realtime_metrics["estimated_accuracy"] * weight
                )

            # 클래스별 성능에 실시간 분포 정보 추가
            if merged.get("class_wise_performance") and realtime_metrics.get("class_distribution"):
                class_perf = merged["class_wise_performance"]
                for class_name, count in realtime_metrics["class_distribution"].items():
                    if class_name in class_perf:
                        # 예측 횟수 정보 추가
                        class_perf[class_name]["prediction_count"] = count
                        class_perf[class_name]["prediction_ratio"] = (
                            count / realtime_metrics["total_predictions"]
                        )

            return PerformanceMetrics(**merged)

        except Exception as e:
            self.logger.warning(f"성능 지표 병합 실패: {e}")
            return base_metrics

    async def get_model_statistics(self) -> Dict[str, Any]:
        """
        모델 사용 통계 조회

        Returns:
            Dict: 상세 사용 통계
        """
        try:
            if not self.model_manager:
                raise RuntimeError("모델 매니저가 초기화되지 않았습니다")

            stats = self.model_manager.statistics

            # 기본 통계
            statistics = {
                "basic_stats": {
                    "total_predictions": stats.total_predictions,
                    "successful_predictions": stats.successful_predictions,
                    "failed_predictions": stats.failed_predictions,
                    "error_rate": stats.error_rate,
                    "success_rate": 1.0 - stats.error_rate if stats.error_rate else 1.0
                },
                "performance_stats": {
                    "average_processing_time": stats.average_processing_time,
                    "uptime_hours": stats.uptime_hours,
                    "last_used": stats.last_used.isoformat() if stats.last_used else None
                },
                "class_prediction_counts": stats.class_prediction_counts,
                "model_info": {
                    "name": self.model_manager.model_name,
                    "version": self.model_manager.model_version,
                    "status": self.model_manager.status.value,
                    "device": str(self.model_manager.device)
                }
            }

            # 시간별 통계 (실제 환경에서는 시계열 데이터베이스 사용)
            statistics["time_based_stats"] = await self._get_time_based_statistics()

            # 성능 트렌드 (최근 N일간의 성능 변화)
            statistics["performance_trends"] = await self._get_performance_trends()

            return statistics

        except Exception as e:
            self.logger.error(f"모델 통계 조회 중 오류: {e}")
            raise

    async def _get_time_based_statistics(self) -> Dict[str, Any]:
        """
        시간 기반 통계 조회 (예시 구현)

        실제 환경에서는 시계열 데이터베이스나 로그 분석을 통해 구현

        Returns:
            Dict: 시간 기반 통계
        """
        # 실제 구현에서는 데이터베이스에서 조회
        # 여기서는 예시 데이터 반환
        now = datetime.now()

        return {
            "hourly_predictions": {
                "current_hour": 45,
                "previous_hour": 38,
                "peak_hour_today": 67,
                "average_per_hour": 42
            },
            "daily_predictions": {
                "today": 520,
                "yesterday": 487,
                "this_week": 3245,
                "this_month": 12890
            },
            "peak_usage_times": [
                {"hour": 9, "average_predictions": 67},
                {"hour": 14, "average_predictions": 58},
                {"hour": 16, "average_predictions": 52}
            ]
        }

    async def _get_performance_trends(self) -> Dict[str, Any]:
        """
        성능 트렌드 분석 (예시 구현)

        Returns:
            Dict: 성능 트렌드 정보
        """
        # 실제 구현에서는 히스토리컬 데이터 분석
        return {
            "accuracy_trend": {
                "last_7_days": [0.94, 0.93, 0.95, 0.94, 0.96, 0.95, 0.94],
                "trend": "stable",
                "change_percentage": 0.2
            },
            "processing_time_trend": {
                "last_7_days": [0.145, 0.142, 0.148, 0.144, 0.146, 0.143, 0.145],
                "trend": "stable",
                "change_percentage": -1.4
            },
            "error_rate_trend": {
                "last_7_days": [0.02, 0.03, 0.01, 0.02, 0.01, 0.02, 0.02],
                "trend": "stable",
                "change_percentage": 0.0
            }
        }

    async def validate_model_health(self) -> Dict[str, Any]:
        """
        모델 상태 검증

        Returns:
            Dict: 모델 상태 검증 결과
        """
        try:
            health_checks = {}
            overall_health = True

            # 1. 모델 로딩 상태 확인
            if self.model_manager.is_loaded:
                health_checks["model_loading"] = {
                    "status": "healthy",
                    "message": "모델이 정상적으로 로드됨"
                }
            else:
                health_checks["model_loading"] = {
                    "status": "unhealthy",
                    "message": "모델이 로드되지 않음"
                }
                overall_health = False

            # 2. 메모리 사용량 확인
            try:
                import psutil
                memory_usage = psutil.virtual_memory().percent
                if memory_usage < 90:
                    health_checks["memory_usage"] = {
                        "status": "healthy",
                        "message": f"메모리 사용률: {memory_usage}%"
                    }
                else:
                    health_checks["memory_usage"] = {
                        "status": "warning",
                        "message": f"높은 메모리 사용률: {memory_usage}%"
                    }
                    if memory_usage > 95:
                        overall_health = False
            except ImportError:
                health_checks["memory_usage"] = {
                    "status": "unknown",
                    "message": "메모리 모니터링 불가능"
                }

            # 3. GPU 상태 확인 (GPU 사용시)
            if settings.USE_GPU:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                        health_checks["gpu_status"] = {
                            "status": "healthy",
                            "message": f"GPU 메모리 사용률: {gpu_memory:.1%}"
                        }
                    else:
                        health_checks["gpu_status"] = {
                            "status": "warning",
                            "message": "GPU를 사용하도록 설정되었으나 사용 불가능"
                        }
                except Exception as e:
                    health_checks["gpu_status"] = {
                        "status": "error",
                        "message": f"GPU 상태 확인 실패: {str(e)}"
                    }

            # 4. 에러율 확인
            if self.model_manager.statistics.total_predictions > 0:
                error_rate = self.model_manager.statistics.error_rate
                if error_rate < 0.05:  # 5% 미만
                    health_checks["error_rate"] = {
                        "status": "healthy",
                        "message": f"에러율: {error_rate:.2%}"
                    }
                elif error_rate < 0.10:  # 10% 미만
                    health_checks["error_rate"] = {
                        "status": "warning",
                        "message": f"높은 에러율: {error_rate:.2%}"
                    }
                else:
                    health_checks["error_rate"] = {
                        "status": "unhealthy",
                        "message": f"매우 높은 에러율: {error_rate:.2%}"
                    }
                    overall_health = False
            else:
                health_checks["error_rate"] = {
                    "status": "unknown",
                    "message": "예측 기록 없음"
                }

            # 5. 응답 시간 확인
            avg_time = self.model_manager.statistics.average_processing_time
            if avg_time and avg_time < 1.0:  # 1초 미만
                health_checks["response_time"] = {
                    "status": "healthy",
                    "message": f"평균 응답 시간: {avg_time:.3f}초"
                }
            elif avg_time and avg_time < 3.0:  # 3초 미만
                health_checks["response_time"] = {
                    "status": "warning",
                    "message": f"느린 응답 시간: {avg_time:.3f}초"
                }
            elif avg_time:
                health_checks["response_time"] = {
                    "status": "unhealthy",
                    "message": f"매우 느린 응답 시간: {avg_time:.3f}초"
                }
                overall_health = False
            else:
                health_checks["response_time"] = {
                    "status": "unknown",
                    "message": "응답 시간 데이터 없음"
                }

            return {
                "overall_health": "healthy" if overall_health else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "checks": health_checks,
                "recommendations": self._generate_health_recommendations(health_checks)
            }

        except Exception as e:
            self.logger.error(f"모델 상태 검증 중 오류: {e}")
            return {
                "overall_health": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "checks": {},
                "recommendations": ["모델 상태 검증 서비스에 문제가 있습니다"]
            }

    def _generate_health_recommendations(self, health_checks: Dict[str, Any]) -> List[str]:
        """
        상태 검증 결과 기반 권장사항 생성

        Args:
            health_checks: 상태 검증 결과

        Returns:
            List[str]: 권장사항 목록
        """
        recommendations = []

        for check_name, check_result in health_checks.items():
            status = check_result.get("status")

            if status == "unhealthy":
                if check_name == "model_loading":
                    recommendations.append("모델을 재로딩하거나 서비스를 재시작하세요")
                elif check_name == "error_rate":
                    recommendations.append("높은 에러율이 감지되었습니다. 모델이나 입력 데이터를 확인하세요")
                elif check_name == "response_time":
                    recommendations.append("응답 시간이 느립니다. GPU 사용을 고려하거나 모델을 최적화하세요")
                elif check_name == "memory_usage":
                    recommendations.append("메모리 사용량이 높습니다. 불필요한 프로세스를 종료하세요")

            elif status == "warning":
                if check_name == "memory_usage":
                    recommendations.append("메모리 사용량을 모니터링하세요")
                elif check_name == "gpu_status":
                    recommendations.append("GPU 설정을 확인하세요")
                elif check_name == "error_rate":
                    recommendations.append("에러율이 증가하고 있습니다. 모니터링을 강화하세요")
                elif check_name == "response_time":
                    recommendations.append("응답 시간을 모니터링하고 최적화를 고려하세요")

        if not recommendations:
            recommendations.append("모든 상태가 정상입니다")

        return recommendations

    def get_service_status(self) -> Dict[str, Any]:
        """
        서비스 전체 상태 정보 반환

        Returns:
            Dict: 서비스 상태 정보
        """
        return {
            "model_service_healthy": self.model_manager is not None,
            "model_loaded": self.model_manager.is_loaded if self.model_manager else False,
            "cache_status": {
                "performance_cache_valid": self._is_performance_cache_valid(),
                "cache_timestamp": self._cache_timestamp.isoformat() if self._cache_timestamp else None
            },
            "supported_operations": [
                "get_model_info",
                "get_model_classes",
                "get_model_performance",
                "reload_model",
                "validate_model_health"
            ]
        }