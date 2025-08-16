from fastapi import APIRouter, HTTPException
from app.services.scheduler_service import simulator_scheduler

router = APIRouter()

@router.get("/status")
async def get_simulator_status():
    """시뮬레이터 상태 조회"""
    status = simulator_scheduler.get_status()  
    # 내부 URL 노출 방지  
    status.pop("backend_service_url", None)  
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
        raise HTTPException(status_code=500, detail=f"시뮬레이터 시작 실패: {str(e)}") from e

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
        logger.exception("시뮬레이터 중지 실패")  
        raise HTTPException(status_code=500, detail="시뮬레이터 중지 실패") from e