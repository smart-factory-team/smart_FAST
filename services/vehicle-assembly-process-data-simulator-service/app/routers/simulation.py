from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import uuid
from datetime import datetime

from app.models.simulation import (
    SimulationRequest, SimulationResponse,
    SimulationStatus, SimulationResult
)
from app.services.simulation import simulations, run_simulation

router = APIRouter()

@router.post("/simulations", response_model=SimulationResponse)
async def create_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """새로운 시뮬레이션 생성 및 시작"""
    simulation_id = str(uuid.uuid4())

    simulations[simulation_id] = {
        "config": request.dict(),
        "status": "pending",
        "created_time": datetime.now()
    }

    background_tasks.add_task(run_simulation, simulation_id, request)

    return SimulationResponse(
        simulation_id=simulation_id,
        status="pending",
        message="시뮬레이션이 시작되었습니다"
    )

@router.get("/simulations", response_model=List[dict])
async def list_simulations():
    """모든 시뮬레이션 목록 조회"""
    return [
        {
            "simulation_id": sim_id,
            "status": sim_data["status"],
            "created_time": sim_data["created_time"],
            "progress": (sim_data.get("processed_images", 0) / sim_data.get("total_images", 1)) * 100
            if sim_data.get("total_images", 0) > 0 else 0
        }
        for sim_id, sim_data in simulations.items()
    ]

@router.get("/simulations/{simulation_id}/status", response_model=SimulationStatus)
async def get_simulation_status(simulation_id: str):
    """시뮬레이션 상태 조회"""
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail="시뮬레이션을 찾을 수 없습니다")

    sim_data = simulations[simulation_id]
    total = sim_data.get("total_images", 0)
    processed = sim_data.get("processed_images", 0)

    return SimulationStatus(
        simulation_id=simulation_id,
        status=sim_data["status"],
        progress=(processed / total * 100) if total > 0 else 0,
        total_images=total,
        processed_images=processed,
        accuracy=sim_data.get("accuracy"),
        start_time=sim_data.get("start_time")
    )

@router.get("/simulations/{simulation_id}/results")
async def get_simulation_results(simulation_id: str):
    """시뮬레이션 결과 조회"""
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail="시뮬레이션을 찾을 수 없습니다")

    sim_data = simulations[simulation_id]

    if sim_data["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="시뮬레이션이 아직 완료되지 않았습니다")

    return {
        "simulation_id": simulation_id,
        "status": sim_data["status"],
        "metrics": sim_data.get("final_metrics", {}),
        "results": sim_data.get("results", []),
        "duration": (sim_data.get("end_time", datetime.now()) - sim_data["start_time"]).total_seconds()
    }

@router.delete("/simulations/{simulation_id}")
async def cancel_simulation(simulation_id: str):
    """시뮬레이션 취소"""
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail="시뮬레이션을 찾을 수 없습니다")

    simulations[simulation_id]["status"] = "cancelled"
    return {"message": "시뮬레이션이 취소되었습니다"}