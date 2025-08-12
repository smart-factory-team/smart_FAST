from fastapi import APIRouter
from app.services.model_client import painting_surface_model_client
from app.services.azure_storage import azure_storage
from app.config.settings import settings
import pytest

router = APIRouter()

pytestmark = pytest.mark.asyncio


@router.post("/azure-storage-connection")
async def test_azure_connection():
    """Azure Storage 연결 테스트"""
    try:
        await azure_storage.connect()
        files = await azure_storage.list_data_files()
        await azure_storage.disconnect()

        return {
            "status": "success",
            "message": "Azure Storage 연결 성공",
            "file_count": len(files),
            "sample_files": files[:5]  # 처음 5개 파일만 표시
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Azure Storage 연결 실패: {str(e)}"
        }


@router.post("/models-connection")
async def test_model_services():
    """도장 표면 결함탐지 서비스 연결 테스트"""
    try:
        is_healthy = await painting_surface_model_client.health_check()

        return {
            "status": "success" if is_healthy else "error",
            "service_name": "painting-surface-defect-detection",
            "healthy": is_healthy,
            "message": "도장 표면 결함탐지 서비스 연결 성공" if is_healthy else "도장 표면 결함탐지 서비스 연결 실패"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"도장 표면 결함탐지 서비스 테스트 실패: {str(e)}"
        }
