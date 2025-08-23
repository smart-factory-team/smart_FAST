from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from services.scheduler_service import scheduler_service
from config.settings import get_simulation_stats, get_settings_summary
from utils.logger import simulator_logger

router = APIRouter(prefix="/simulator", tags=["Simulator Control"])

class SimulationSettings(BaseModel):
    """시뮬레이션 설정 모델"""
    interval_seconds: Optional[int] = None
    start_inspection_id: Optional[int] = None

@router.get("/status", response_model=Dict[str, Any])
async def get_simulator_status():
    """시뮬레이터 상태 조회"""
    try:
        # 스케줄러 상태
        scheduler_status = scheduler_service.get_scheduler_status()
        
        # 설정 정보
        settings_summary = get_settings_summary()
        
        # 헬스체크
        health_status = await scheduler_service.health_check()
        
        return {
            "simulator_status": "running" if scheduler_status['scheduler_info']['running'] else "stopped",
            "scheduler": scheduler_status,
            "health": health_status,
            "settings": settings_summary,
            "timestamp": scheduler_status['scheduler_info'].get('last_execution')
        }
    
    except Exception as e:
        simulator_logger.logger.error(f"시뮬레이터 상태 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")

@router.post("/start", response_model=Dict[str, Any])
async def start_simulation():
    """시뮬레이션 시작"""
    try:
        simulator_logger.logger.info("🚀 시뮬레이션 시작 요청")
        
        # 스케줄러 시작
        success = await scheduler_service.start_scheduler()
        
        if success:
            status = scheduler_service.get_scheduler_status()
            return {
                "status": "success",
                "message": "시뮬레이션이 성공적으로 시작되었습니다",
                "scheduler_info": status['scheduler_info'],
                "next_execution": status['scheduler_info'].get('next_execution')
            }
        else:
            return {
                "status": "failed",
                "message": "시뮬레이션 시작에 실패했습니다. 서비스 연결을 확인해주세요."
            }
    
    except Exception as e:
        simulator_logger.logger.error(f"시뮬레이션 시작 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"시뮬레이션 시작 실패: {str(e)}")

@router.post("/stop", response_model=Dict[str, Any])
async def stop_simulation():
    """시뮬레이션 중지"""
    try:
        simulator_logger.logger.info("⏹️ 시뮬레이션 중지 요청")
        
        # 스케줄러 중지
        success = await scheduler_service.stop_scheduler()
        
        if success:
            # 최종 통계
            final_stats = get_simulation_stats()
            
            return {
                "status": "success",
                "message": "시뮬레이션이 성공적으로 중지되었습니다",
                "final_statistics": final_stats
            }
        else:
            return {
                "status": "failed",
                "message": "시뮬레이션 중지에 실패했습니다"
            }
    
    except Exception as e:
        simulator_logger.logger.error(f"시뮬레이션 중지 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"시뮬레이션 중지 실패: {str(e)}")

@router.post("/restart", response_model=Dict[str, Any])
async def restart_simulation():
    """시뮬레이션 재시작"""
    try:
        simulator_logger.logger.info("🔄 시뮬레이션 재시작 요청")
        
        # 스케줄러 재시작
        success = await scheduler_service.restart_scheduler()
        
        if success:
            status = scheduler_service.get_scheduler_status()
            return {
                "status": "success",
                "message": "시뮬레이션이 성공적으로 재시작되었습니다",
                "scheduler_info": status['scheduler_info'],
                "next_execution": status['scheduler_info'].get('next_execution')
            }
        else:
            return {
                "status": "failed",
                "message": "시뮬레이션 재시작에 실패했습니다"
            }
    
    except Exception as e:
        simulator_logger.logger.error(f"시뮬레이션 재시작 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"시뮬레이션 재시작 실패: {str(e)}")

@router.post("/execute/manual", response_model=Dict[str, Any])
async def execute_manual_simulation():
    """수동 시뮬레이션 실행 (테스트용)"""
    try:
        simulator_logger.logger.info("🔧 수동 시뮬레이션 실행 요청")
        
        # 수동 실행
        result = await scheduler_service.manual_execution()
        
        if result['success']:
            return {
                "status": "success",
                "message": f"inspection_{result['inspection_id']:03d} 수동 실행 완료",
                "execution_result": result,
                "timestamp": result['timestamp']
            }
        else:
            return {
                "status": "failed",
                "message": "수동 실행 실패",
                "error": result.get('error'),
                "execution_result": result
            }
    
    except Exception as e:
        simulator_logger.logger.error(f"수동 실행 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"수동 실행 실패: {str(e)}")

@router.get("/statistics", response_model=Dict[str, Any])
async def get_simulation_statistics():
    """시뮬레이션 통계 조회"""
    try:
        # 기본 통계
        basic_stats = get_simulation_stats()
        
        # 스케줄러 상태에서 추가 정보
        scheduler_status = scheduler_service.get_scheduler_status()
        
        return {
            "simulation_statistics": basic_stats,
            "execution_info": {
                "total_executions": scheduler_status['scheduler_info']['execution_count'],
                "average_processing_time": scheduler_status['scheduler_info']['average_processing_time'],
                "start_time": scheduler_status['scheduler_info']['start_time'],
                "last_execution": scheduler_status['scheduler_info']['last_execution']
            },
            "current_status": {
                "running": scheduler_status['scheduler_info']['running'],
                "current_inspection_id": basic_stats['current_inspection_id'],
                "next_execution": scheduler_status['scheduler_info']['next_execution']
            }
        }
    
    except Exception as e:
        simulator_logger.logger.error(f"통계 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@router.get("/logs", response_model=Dict[str, Any])
async def get_recent_logs(
    log_type: str = Query("all", description="로그 타입: all, results, errors, anomalies"),
    limit: int = Query(50, ge=1, le=500, description="조회할 로그 수 (1-500)")
):
    """최근 로그 조회"""
    try:
        if log_type not in ["all", "results", "errors", "anomalies"]:
            raise HTTPException(status_code=400, detail="log_type은 all, results, errors, anomalies 중 하나여야 합니다")
        
        # 로그 조회
        logs = simulator_logger.get_recent_logs(log_type=log_type, limit=limit)
        
        # 로그 타입별 카운트
        log_counts = {
            "total": len(logs),
            "results": len([log for log in logs if log.get('log_type') == 'result']),
            "errors": len([log for log in logs if log.get('log_type') == 'error']),
            "anomalies": len([log for log in logs if log.get('log_type') == 'anomaly'])
        }
        
        return {
            "log_type": log_type,
            "limit": limit,
            "log_counts": log_counts,
            "logs": logs
        }
    
    except HTTPException:
        raise
    except Exception as e:
        simulator_logger.logger.error(f"로그 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"로그 조회 실패: {str(e)}")

@router.get("/logs/summary", response_model=Dict[str, Any])
async def get_logs_summary():
    """로그 요약 정보"""
    try:
        # 각 타입별로 최근 로그 조회
        recent_results = simulator_logger.get_recent_logs(log_type="results", limit=10)
        recent_errors = simulator_logger.get_recent_logs(log_type="errors", limit=10)
        recent_anomalies = simulator_logger.get_recent_logs(log_type="anomalies", limit=10)
        
        # 요약 통계
        summary = {
            "recent_activity": {
                "latest_result": recent_results[0] if recent_results else None,
                "latest_error": recent_errors[0] if recent_errors else None,
                "latest_anomaly": recent_anomalies[0] if recent_anomalies else None
            },
            "counts": {
                "recent_results": len(recent_results),
                "recent_errors": len(recent_errors),
                "recent_anomalies": len(recent_anomalies)
            },
            "health_indicators": {
                "has_recent_errors": len(recent_errors) > 0,
                "has_recent_anomalies": len(recent_anomalies) > 0,
                "error_rate": len(recent_errors) / max(1, len(recent_results) + len(recent_errors)) * 100
            }
        }
        
        return summary
    
    except Exception as e:
        simulator_logger.logger.error(f"로그 요약 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"로그 요약 조회 실패: {str(e)}")

@router.get("/health", response_model=Dict[str, Any])
async def get_health_status():
    """헬스체크"""
    try:
        health_status = await scheduler_service.health_check()
        
        # 추가 상태 정보
        scheduler_status = scheduler_service.get_scheduler_status()
        
        return {
            "overall_health": health_status['healthy'],
            "components": {
                "scheduler": health_status['scheduler_running'],
                "azure_storage": health_status['azure_storage'],
                "model_service": health_status['model_service'],
                "initialization": health_status['initialization_completed']
            },
            "details": health_status,
            "runtime_info": {
                "running": scheduler_status['scheduler_info']['running'],
                "execution_count": scheduler_status['scheduler_info']['execution_count'],
                "last_execution": scheduler_status['scheduler_info']['last_execution']
            }
        }
    
    except Exception as e:
        simulator_logger.logger.error(f"헬스체크 실패: {str(e)}")
        return {
            "overall_health": False,
            "error": str(e),
            "components": {
                "scheduler": False,
                "azure_storage": False,
                "model_service": False,
                "initialization": False
            }
        }

@router.get("/settings", response_model=Dict[str, Any])
async def get_current_settings():
    """현재 설정 조회"""
    try:
        settings_summary = get_settings_summary()
        simulation_stats = get_simulation_stats()
        
        return {
            "application_settings": settings_summary,
            "current_simulation": simulation_stats,
            "runtime_status": {
                "scheduler_running": scheduler_service.running,
                "initialization_completed": scheduler_service.initialization_completed
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 조회 실패: {str(e)}")