from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.services.scheduler_service import scheduler_service
from app.utils.logger import system_log

router = APIRouter(prefix="/simulator", tags=["simulator"])


class SimulationResponse(BaseModel):
    """시뮬레이션 응답 모델"""

    success: bool
    message: str
    data: Dict[str, Any] = {}


@router.post("/start", response_model=SimulationResponse)
async def start_simulation():
    """시뮬레이션 시작"""
    try:
        success = await scheduler_service.start_simulation()

        if success:
            status = scheduler_service.get_simulation_status()
            return SimulationResponse(
                success=True,
                message="시뮬레이션이 성공적으로 시작되었습니다.",
                data=status,
            )
        else:
            return SimulationResponse(
                success=False,
                message="시뮬레이션 시작에 실패했습니다. (이미 실행 중이거나 API 서버 연결 실패)",
                data={},
            )

    except Exception as e:
        system_log.error(f"시뮬레이션 시작 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")


@router.post("/stop", response_model=SimulationResponse)
async def stop_simulation():
    """시뮬레이션 종료"""
    try:
        success = await scheduler_service.stop_simulation()

        if success:
            return SimulationResponse(
                success=True, message="시뮬레이션이 성공적으로 종료되었습니다.", data={}
            )
        else:
            return SimulationResponse(
                success=False,
                message="시뮬레이션 종료에 실패했습니다. (실행 중이 아님)",
                data={},
            )

    except Exception as e:
        system_log.error(f"시뮬레이션 종료 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")


@router.get("/status", response_model=SimulationResponse)
async def get_simulation_status():
    """시뮬레이션 상태 조회"""
    try:
        status = scheduler_service.get_simulation_status()

        return SimulationResponse(
            success=True,
            message="시뮬레이션 상태를 성공적으로 조회했습니다.",
            data=status,
        )

    except Exception as e:
        system_log.error(f"시뮬레이션 상태 조회 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")
