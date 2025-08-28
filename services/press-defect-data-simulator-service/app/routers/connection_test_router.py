from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio

from app.services.azure_storage import azure_storage_service
from app.services.model_client import model_service_client
from app.utils.logger import simulator_logger

router = APIRouter(prefix="/connection", tags=["Connection Test"])

async def ensure_azure_initialized():
    """Azure Storage Service 초기화 보장"""
    if not azure_storage_service.connection_tested:
        simulator_logger.logger.info("🔄 Azure Storage Service 초기화 중...")
        success = await azure_storage_service.initialize()
        if success:
            simulator_logger.logger.info("✅ Azure Storage Service 초기화 완료")
        else:
            simulator_logger.logger.warning("⚠️ Azure Storage Service 초기화 실패")
        return success
    return True

async def ensure_model_service_connected():
    """Model Service 연결 상태 보장"""
    # 실제 연결 테스트를 통해 service_available 플래그 업데이트
    connection_result = await model_service_client.test_connection()
    return connection_result

@router.get("/test/azure", response_model=Dict[str, Any])
async def test_azure_connection():
    """Azure Storage 연결 테스트"""
    try:
        simulator_logger.logger.info("🔍 Azure Storage 연결 테스트 시작")
        
        # 자동 초기화 보장
        await ensure_azure_initialized()
        
        # 연결 테스트
        connection_success = await azure_storage_service.test_connection()
        
        # 서비스 상태 조회
        service_status = azure_storage_service.get_service_status()
        
        if connection_success:
            # 추가 정보 조회
            try:
                inspection_list = await azure_storage_service.get_inspection_list()
                available_inspections = len(inspection_list)
                
                # 첫 번째 inspection 검증 (있는 경우)
                validation_result = None
                if inspection_list:
                    first_inspection_id = int(inspection_list[0].split('_')[1])
                    validation_result = await azure_storage_service.validate_inspection_completeness(first_inspection_id)
                
                return {
                    "status": "success",
                    "message": "Azure Storage 연결 성공",
                    "service_status": service_status,
                    "available_inspections": available_inspections,
                    "inspection_list": inspection_list[:10],  # 처음 10개만
                    "sample_validation": validation_result
                }
            
            except Exception as detail_error:
                return {
                    "status": "partial_success",
                    "message": "Azure Storage 연결 성공하지만 상세 정보 조회 실패",
                    "service_status": service_status,
                    "detail_error": str(detail_error)
                }
        else:
            return {
                "status": "failed",
                "message": "Azure Storage 연결 실패",
                "service_status": service_status
            }
    
    except Exception as e:
        simulator_logger.logger.error(f"Azure 연결 테스트 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Azure 연결 테스트 실패: {str(e)}")

@router.get("/test/model", response_model=Dict[str, Any])
async def test_model_service():
    """모델 서비스 연결 테스트"""
    try:
        simulator_logger.logger.info("🔍 모델 서비스 연결 테스트 시작")
        
        # 🔧 핵심 수정: 먼저 test_connection을 호출해서 service_available 플래그 업데이트
        basic_connection = await model_service_client.test_connection()
        
        if not basic_connection:
            return {
                "status": "failed",
                "message": "모델 서비스 기본 연결 실패",
                "client_status": model_service_client.get_client_status(),
                "service_details": None
            }
        
        # 연결 성공 후 종합 서비스 체크
        all_services_result = await model_service_client.check_all_services()
        
        # 클라이언트 상태 (이제 service_available이 True로 업데이트됨)
        client_status = model_service_client.get_client_status()
        
        return {
            "status": "success" if all_services_result['overall_status'] == 'healthy' else "failed",
            "message": f"모델 서비스 상태: {all_services_result['overall_status']}",
            "client_status": client_status,
            "service_details": all_services_result
        }
    
    except Exception as e:
        simulator_logger.logger.error(f"모델 서비스 연결 테스트 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"모델 서비스 연결 테스트 실패: {str(e)}")

@router.get("/test/all", response_model=Dict[str, Any])
async def test_all_connections():
    """모든 외부 서비스 연결 테스트"""
    try:
        simulator_logger.logger.info("🔍 전체 서비스 연결 테스트 시작")
        
        # 서비스 초기화 및 연결 보장
        await ensure_azure_initialized()
        await ensure_model_service_connected()
        
        # 병렬 테스트 실행
        azure_task = azure_storage_service.test_connection()
        model_task = model_service_client.test_connection()
        
        azure_success, model_success = await asyncio.gather(azure_task, model_task)
        
        # 상세 정보 수집
        azure_status = azure_storage_service.get_service_status()
        model_status = model_service_client.get_client_status()
        
        # 전체 상태 판정
        overall_success = azure_success and model_success
        
        return {
            "status": "success" if overall_success else "partial_failure",
            "message": f"전체 연결 테스트 {'성공' if overall_success else '일부 실패'}",
            "results": {
                "azure_storage": {
                    "connected": azure_success,
                    "status": azure_status
                },
                "model_service": {
                    "connected": model_success,
                    "status": model_status
                }
            },
            "summary": {
                "total_services": 2,
                "connected_services": sum([azure_success, model_success]),
                "failed_services": 2 - sum([azure_success, model_success])
            }
        }
    
    except Exception as e:
        simulator_logger.logger.error(f"전체 연결 테스트 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"전체 연결 테스트 실패: {str(e)}")

@router.get("/status/azure", response_model=Dict[str, Any])
async def get_azure_status():
    """Azure Storage 상태 조회"""
    try:
        # 초기화 보장
        await ensure_azure_initialized()
        
        service_status = azure_storage_service.get_service_status()
        
        # 추가 상태 정보
        if service_status['connected']:
            try:
                inspection_list = await azure_storage_service.get_inspection_list()
                service_status['available_inspections'] = len(inspection_list)
                service_status['inspection_range'] = {
                    "first": inspection_list[0] if inspection_list else None,
                    "last": inspection_list[-1] if inspection_list else None
                }
            except:
                service_status['additional_info_error'] = "상세 정보 조회 실패"
        
        return service_status
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure 상태 조회 실패: {str(e)}")

@router.get("/status/model", response_model=Dict[str, Any])
async def get_model_status():
    """모델 서비스 상태 조회"""
    try:
        client_status = model_service_client.get_client_status()
        
        # 🔧 핵심 수정: service_available이 False인 경우 연결 테스트 재시도
        if not client_status.get('service_available', False):
            simulator_logger.logger.info("🔄 Model Service 연결 상태 재확인 중...")
            connection_result = await model_service_client.test_connection()
            if connection_result:
                # 연결 성공하면 업데이트된 상태 다시 조회
                client_status = model_service_client.get_client_status()
                simulator_logger.logger.info("✅ Model Service 연결 상태 업데이트 완료")
        
        # 실시간 헬스체크 추가 (service_available이 True인 경우에만)
        if client_status.get('service_available', False):
            try:
                health_result = await model_service_client.check_service_health()
                client_status['current_health'] = health_result
            except Exception as health_error:
                simulator_logger.logger.warning(f"헬스체크 실패: {str(health_error)}")
                client_status['health_check_error'] = str(health_error)
        
        return client_status
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 서비스 상태 조회 실패: {str(e)}")

@router.post("/validate/inspection/{inspection_id}", response_model=Dict[str, Any])
async def validate_inspection_data(inspection_id: int):
    """특정 inspection 데이터 검증"""
    try:
        if not (1 <= inspection_id <= 79):
            raise HTTPException(status_code=400, detail="inspection_id는 1-79 범위여야 합니다.")
        
        simulator_logger.logger.info(f"🔍 inspection_{inspection_id:03d} 데이터 검증 시작")
        
        # Azure Storage 초기화 및 연결 확인
        await ensure_azure_initialized()
        
        if not await azure_storage_service.test_connection():
            raise HTTPException(status_code=503, detail="Azure Storage 연결 실패")
        
        # 데이터 완성도 검사
        validation_result = await azure_storage_service.validate_inspection_completeness(inspection_id)
        
        # 이미지 목록 조회
        image_list = await azure_storage_service.get_inspection_images(inspection_id)
        
        return {
            "inspection_id": inspection_id,
            "validation_result": validation_result,
            "image_details": image_list,
            "status": "complete" if validation_result.get('is_complete', False) else "incomplete"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        simulator_logger.logger.error(f"inspection_{inspection_id:03d} 검증 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터 검증 실패: {str(e)}")