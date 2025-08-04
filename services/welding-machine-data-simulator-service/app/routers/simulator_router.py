from fastapi import APIRouter, HTTPException
from app.services.model_client import model_client
from app.services.scheduler_service import simulator_scheduler
from app.config.settings import settings

import os

router = APIRouter()


@router.get("/status")
async def get_simulator_status():
    """시뮬레이터 상태 조회"""
    status = simulator_scheduler.get_status()

    # 모델 서비스 헬스 체크 추가
    if simulator_scheduler.is_running:
        health_status = await model_client.health_check_all()
        status["model_services_health"] = health_status

    return status


@router.post("/start")
async def start_simulator():
    """시뮬레이션 시작"""
    try:
        await simulator_scheduler.start()
        return {
            "message": "시뮬레이터가 시작되었습니다.",
            "status": simulator_scheduler.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시뮬레이터 시작 실패: {str(e)}")


@router.post("/stop")
async def stop_simulator():
    """시뮬레이션 중지"""
    try:
        await simulator_scheduler.stop()
        return {
            "message": "시뮬레이터가 중지되었습니다.",
            "status": simulator_scheduler.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시뮬레이터 중지 실패: {str(e)}")


@router.get("/logs/recent")
async def get_recent_logs():
    """최근 로그 조회"""
    try:
        log_file_path = os.path.join(
            settings.log_directory, settings.log_filename)

        if not os.path.exists(log_file_path):
            return {"logs": [], "message": "로그 파일이 없습니다."}

        # 최근 10개 로그만 반환
        logs = []
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-10:]:  # 최근 10개
                try:
                    import json
                    logs.append(json.loads(line.strip()))
                except:
                    continue

        return {
            "logs": logs,
            "total_count": len(lines) if 'lines' in locals() else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"로그 조회 실패: {str(e)}")
