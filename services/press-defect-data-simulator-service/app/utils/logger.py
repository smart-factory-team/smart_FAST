import os
import logging
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import settings

class SimulatorLogger:
    """시뮬레이터 전용 로거"""
    
    def __init__(self):
        self.setup_log_directory()
        self.logger = self.setup_logger()
        
    def setup_log_directory(self):
        """로그 디렉토리 생성"""
        log_dir = Path(settings.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # 하위 디렉토리들 생성
        (log_dir / "simulation").mkdir(exist_ok=True)
        (log_dir / "errors").mkdir(exist_ok=True)
        (log_dir / "results").mkdir(exist_ok=True)
        
    def setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger("simulator")
        logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러 (일반 로그)
        file_handler = RotatingFileHandler(
            os.path.join(settings.log_dir, settings.log_file_name),
            maxBytes=settings.log_max_size,
            backupCount=settings.log_backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 에러 전용 파일 핸들러
        error_handler = RotatingFileHandler(
            os.path.join(settings.log_dir, "errors", "error.log"),
            maxBytes=settings.log_max_size,
            backupCount=settings.log_backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        return logger
    
    def log_simulation_start(self, inspection_id: int):
        """시뮬레이션 시작 로그"""
        message = f"🚀 시뮬레이션 시작: inspection_{inspection_id:03d}"
        self.logger.info(message)
        
    def log_simulation_success(self, inspection_id: int, result: Dict[str, Any], processing_time: float):
        """시뮬레이션 성공 로그 (콘솔만)"""
        quality_status = result.get('final_judgment', {}).get('quality_status', 'Unknown')
        recommendation = result.get('final_judgment', {}).get('recommendation', 'Unknown')
        
        # 콘솔에만 출력 (정상 처리)
        message = f"✅ inspection_{inspection_id:03d} 완료: {quality_status} ({recommendation}) - {processing_time:.2f}초"
        self.logger.info(message)
        
        # 결과 로그 파일에는 상세 정보 저장
        self._save_result_log(inspection_id, result, processing_time, success=True)
    
    def log_simulation_failure(self, inspection_id: int, error: str, processing_time: Optional[float] = None):
        """시뮬레이션 실패 로그 (파일과 콘솔 모두)"""
        time_info = f" - {processing_time:.2f}초" if processing_time else ""
        message = f"❌ inspection_{inspection_id:03d} 실패: {error}{time_info}"
        
        # 에러는 파일과 콘솔 모두에 기록
        self.logger.error(message)
        
        # 에러 상세 로그 저장
        self._save_error_log(inspection_id, error, processing_time)
    
    def log_anomaly_detection(self, inspection_id: int, result: Dict[str, Any]):
        """이상 감지 결과 로그 (파일과 콘솔 모두)"""
        quality_status = result.get('final_judgment', {}).get('quality_status', 'Unknown')
        missing_holes = result.get('final_judgment', {}).get('missing_holes', [])
        
        if quality_status == "결함품":
            message = f"🚨 결함품 감지: inspection_{inspection_id:03d} - 누락된 구멍: {missing_holes}"
            self.logger.warning(message)
            
            # 이상 감지 로그 파일에 저장
            self._save_anomaly_log(inspection_id, result)
        
    def log_scheduler_event(self, event: str, details: Optional[Dict] = None):
        """스케줄러 이벤트 로그"""
        message = f"📅 스케줄러: {event}"
        if details:
            message += f" - {details}"
        self.logger.info(message)
    
    def log_azure_connection(self, success: bool, details: Optional[str] = None):
        """Azure 연결 로그"""
        if success:
            message = f"☁️ Azure Storage 연결 성공"
            if details:
                message += f" - {details}"
            self.logger.info(message)
        else:
            message = f"❌ Azure Storage 연결 실패"
            if details:
                message += f" - {details}"
            self.logger.error(message)
    
    def log_model_service(self, success: bool, details: Optional[str] = None):
        """모델 서비스 연결 로그"""
        if success:
            message = f"🤖 모델 서비스 연결 성공"
            if details:
                message += f" - {details}"
            self.logger.info(message)
        else:
            message = f"❌ 모델 서비스 연결 실패"
            if details:
                message += f" - {details}"
            self.logger.error(message)
    
    def _save_result_log(self, inspection_id: int, result: Dict[str, Any], processing_time: float, success: bool):
        """결과 로그 파일 저장"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "inspection_id": f"inspection_{inspection_id:03d}",
            "success": success,
            "processing_time": processing_time,
            "result": result
        }
        
        log_file = os.path.join(settings.log_dir, "results", f"results_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"결과 로그 저장 실패: {e}")
    
    def _save_error_log(self, inspection_id: int, error: str, processing_time: Optional[float]):
        """에러 로그 파일 저장"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "inspection_id": f"inspection_{inspection_id:03d}",
            "error": error,
            "processing_time": processing_time
        }
        
        log_file = os.path.join(settings.log_dir, "errors", f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"에러 로그 저장 실패: {e}")
    
    def _save_anomaly_log(self, inspection_id: int, result: Dict[str, Any]):
        """이상 감지 로그 파일 저장"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "inspection_id": f"inspection_{inspection_id:03d}",
            "anomaly_type": "defect_detected",
            "quality_status": result.get('final_judgment', {}).get('quality_status'),
            "missing_holes": result.get('final_judgment', {}).get('missing_holes', []),
            "recommendation": result.get('final_judgment', {}).get('recommendation'),
            "quality_inspection": result.get('quality_inspection', {})
        }
        
        log_file = os.path.join(settings.log_dir, "simulation", f"anomalies_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"이상 감지 로그 저장 실패: {e}")
    
    def get_recent_logs(self, log_type: str = "all", limit: int = 100) -> list:
        """최근 로그 조회"""
        logs = []
        
        try:
            if log_type in ["all", "results"]:
                result_file = os.path.join(settings.log_dir, "results", f"results_{datetime.now().strftime('%Y%m%d')}.jsonl")
                if os.path.exists(result_file):
                    with open(result_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines[-limit:]:
                            try:
                                log_data = json.loads(line.strip())
                                log_data["log_type"] = "result"
                                logs.append(log_data)
                            except json.JSONDecodeError:
                                continue
            
            if log_type in ["all", "errors"]:
                error_file = os.path.join(settings.log_dir, "errors", f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl")
                if os.path.exists(error_file):
                    with open(error_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines[-limit:]:
                            try:
                                log_data = json.loads(line.strip())
                                log_data["log_type"] = "error"
                                logs.append(log_data)
                            except json.JSONDecodeError:
                                continue
            
            if log_type in ["all", "anomalies"]:
                anomaly_file = os.path.join(settings.log_dir, "simulation", f"anomalies_{datetime.now().strftime('%Y%m%d')}.jsonl")
                if os.path.exists(anomaly_file):
                    with open(anomaly_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines[-limit:]:
                            try:
                                log_data = json.loads(line.strip())
                                log_data["log_type"] = "anomaly"
                                logs.append(log_data)
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            self.logger.error(f"로그 조회 실패: {e}")
        
        # 시간순 정렬
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return logs[:limit]

# 전역 로거 인스턴스
simulator_logger = SimulatorLogger()