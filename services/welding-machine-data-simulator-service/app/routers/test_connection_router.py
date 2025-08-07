from fastapi import APIRouter
from app.services.model_client import model_client
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
    """모델 서비스 연결 테스트"""
    try:
        health_status = await model_client.health_check_all()

        healthy_services = [name for name,
                            status in health_status.items() if status]
        unhealthy_services = [name for name,
                              status in health_status.items() if not status]

        return {
            "status": "success" if healthy_services else "error",
            "healthy_services": healthy_services,
            "unhealthy_services": unhealthy_services,
            "total_services": len(settings.model_services),
            "healthy_count": len(healthy_services)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"모델 서비스 테스트 실패: {str(e)}"
        }
