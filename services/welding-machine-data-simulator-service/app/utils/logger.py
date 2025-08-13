import json
import os
from datetime import datetime
from typing import Dict, Any
from app.config.settings import settings


class AnomalyLogger:
    def __init__(self):
        # 로그 디렉토리 생성
        os.makedirs(settings.log_directory, exist_ok=True)
        self.log_file_path = os.path.join(
            settings.log_directory, settings.log_filename)

    def log_anomaly(self, service_name: str, prediction_result: Dict[str, Any], original_data: Dict[str, Any]):
        """이상 감지 결과를 로그 파일에 저장"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "service_name": service_name,
            "prediction": prediction_result,
            "original_data": original_data
        }

        # JSON 파일에 추가
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # 콘솔 출력
        print(f"🚨 ANOMALY DETECTED: {service_name}")
        print(f"   └─ MAE: {prediction_result.get('mae', 'N/A')}")
        print(f"   └─ Threshold: {prediction_result.get('threshold', 'N/A')}")
        print(f"   └─ Status: {prediction_result.get('status', 'N/A')}")
        print(f"   └─ Time: {log_entry['timestamp']}")
        print("-" * 50)

    def log_normal_processing(self, service_name: str, prediction_result: Dict[str, Any]):
        """정상 처리 결과를 콘솔에만 출력"""
        print(
            f"✅ NORMAL: {service_name} - MAE: {prediction_result.get('mae', 'N/A')}")

    def log_error(self, service_name: str, error_message: str, original_data: Dict[str, Any] = None):
        """에러 로그"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "service_name": service_name,
            "error": error_message,
            "original_data": original_data
        }

        error_log_path = os.path.join(
            settings.log_directory, settings.error_log_filename)
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"❌ ERROR: {service_name} - {error_message}")


# 글로벌 로거 인스턴스
anomaly_logger = AnomalyLogger()
