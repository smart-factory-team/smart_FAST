import os
from typing import List, Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ✅ 환경 구분 (기본값을 local로 변경)
    environment: str = "production"  # 기본값을 다시 local로 변경

    # Azure Storage 설정
    azure_connection_string: str
    azure_container_name: str = "simulator-data"

    # Welding Machine 전용 설정
    welding_data_folder: str = "welding-machine"

    # 스케줄러 설정
    scheduler_interval_minutes: int = 1
    batch_size: int = 10

    # ✅ 기존 호환성을 위한 필드 (deprecated but needed)
    welding_machine_url: str = "http://localhost:8006"

    # ✅ 환경별 서비스 URL 설정
    # 로컬 개발
    local_gateway_url: str = "http://localhost:8088"
    local_model_service_url: str = "http://localhost:8006"

    # Docker Compose
    docker_gateway_url: str = "http://gateway:8088"
    docker_model_service_url: str = "http://welding-model-service:8006"

    # AKS Kubernetes (내부 서비스 통신)
    kubernetes_gateway_url: str = "http://gateway.smart-be.svc.cluster.local:8088"
    kubernetes_model_service_url: str = "http://sf-welding-machine-defect-detection-model-service.sf-welding-machine-defect-detection-model-service.svc.cluster.local:80"

    # ✅ Production (현재 실제 배포된 URL) - 기본값으로 설정
    production_gateway_url: str = "http://20.249.138.42:8088"  # 배포시 기본값
    production_model_service_url: str = "http://20.249.138.42:8088/api/v1/welding/defect"

    # HTTP 설정
    spring_boot_timeout: int = 30
    spring_boot_max_retries: int = 3
    http_timeout: int = 30
    max_retries: int = 3

    # 로그 설정
    log_directory: str = "logs"
    log_filename: str = "welding_anomaly_detections.json"
    error_log_filename: str = "welding_errors.json"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ✅ 명시적으로 environment가 설정되지 않은 경우에만 자동 감지
        if not kwargs.get('environment') and not os.getenv('ENVIRONMENT'):
            # 특정 조건에서만 production으로 감지 (더 엄격한 조건)
            if (hasattr(self, 'azure_connection_string') and
                self.azure_connection_string and
                self.azure_connection_string != "your_azure_connection_string" and
                "localhost" not in self.azure_connection_string and
                # 추가 조건: Kubernetes 환경 변수 존재 여부
                    (os.getenv('KUBERNETES_SERVICE_HOST') or os.getenv('WEBSITE_HOSTNAME'))):
                self.environment = "production"
            else:
                self.environment = "local"

    @property
    def gateway_service_url(self) -> str:
        """환경에 따른 게이트웨이 서비스 URL"""
        url_map = {
            "local": self.local_gateway_url,
            "development": self.local_gateway_url,
            "docker": self.docker_gateway_url,
            "kubernetes": self.kubernetes_gateway_url,
            "production": self.production_gateway_url,
        }
        return url_map.get(self.environment, self.local_gateway_url)

    @property
    def model_service_url(self) -> str:
        """환경에 따른 모델 서빙 서비스 URL"""
        url_map = {
            "local": self.local_model_service_url,
            "development": self.local_model_service_url,
            "docker": self.docker_model_service_url,
            "kubernetes": self.kubernetes_model_service_url,
            "production": self.production_model_service_url,
        }
        return url_map.get(self.environment, self.local_model_service_url)

    @property
    def model_services(self) -> Dict[str, str]:
        """Welding Machine 모델 서비스 URL (기존 호환성 유지)"""
        return {
            "welding-machine": self.welding_machine_url  # 기존 필드 사용
        }

    @property
    def spring_boot_service_url(self) -> str:
        """기존 호환성을 위한 별칭"""
        return self.gateway_service_url

    @property
    def spring_boot_endpoints(self) -> Dict[str, str]:
        """스프링부트 서비스 엔드포인트 (게이트웨이 경유)"""
        base_url = self.gateway_service_url

        return {
            "welding_data": f"{base_url}/weldingMachineDefectDetectionLogs",
            "health": f"{base_url}/actuator/health",
            "status": f"{base_url}/weldingMachineDefectDetectionLogs"
        }

    @property
    def is_local_environment(self) -> bool:
        """로컬 환경 여부 확인"""
        return self.environment in ["local", "development"]

    @property
    def is_production_environment(self) -> bool:
        """프로덕션 환경 여부 확인"""
        return self.environment in ["production", "kubernetes"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


settings = Settings()
