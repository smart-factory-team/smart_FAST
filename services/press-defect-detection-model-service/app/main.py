from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
import asyncio

# 모델 import 추가
from press_models.yolo_model import YOLOv7Model

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Press Defect Detection API",
    description="자동차 부품 프레스 구멍 결함 탐지 AI 서비스",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수 - 모델 인스턴스
yolo_model = YOLOv7Model()
model_loaded = False
model_loading_error = None
service_start_time = datetime.now()

# 애플리케이션 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """서비스 시작 시 실행되는 함수"""
    global model_loaded, model_loading_error, yolo_model
    
    logger.info("🚀 Press Defect Detection API 서비스 시작")
    logger.info("⏳ AI 모델 로딩 중...")
    
    try:
        # 실제 모델 로딩
        await yolo_model.load_model()
        model_loaded = True
        logger.info("✅ AI 모델 로딩 완료")
        
    except Exception as e:
        model_loading_error = str(e)
        model_loaded = False
        logger.error(f"❌ AI 모델 로딩 실패: {e}")

# 애플리케이션 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    """서비스 종료 시 실행되는 함수"""
    logger.info("🛑 Press Defect Detection API 서비스 종료")

# 기본 엔드포인트들
@app.get("/")
async def root():
    """서비스 정보 반환"""
    return {
        "service": "Press Defect Detection API",
        "version": "1.0.0",
        "description": "자동차 부품 프레스 구멍 결함 탐지 AI 서비스",
        "status": "running",
        "start_time": service_start_time.isoformat(),
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "ready": "GET /ready", 
            "startup": "GET /startup",
            "predict": "POST /predict",
            "predict_file": "POST /predict/file",
            "model_info": "GET /model/info"
        }
    }

@app.get("/health")
async def health_check():
    """기본 헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - service_start_time).total_seconds()
    }

@app.get("/ready")
async def readiness_check():
    """준비 상태 체크 - AI 모델이 로딩되었는지 확인"""
    global model_loaded, model_loading_error
    
    if model_loaded:
        return {
            "status": "ready",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat()
        }
    elif model_loading_error:
        raise HTTPException(
            status_code=503, 
            detail={
                "status": "not_ready",
                "model_loaded": False,
                "error": model_loading_error,
                "timestamp": datetime.now().isoformat()
            }
        )
    else:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "loading",
                "model_loaded": False,
                "message": "AI 모델 로딩 중입니다...",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/startup")
async def startup_check():
    """서비스 시작 준비 상태 체크"""
    return {
        "status": "started",
        "service_ready": True,
        "model_ready": model_loaded,
        "start_time": service_start_time.isoformat(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """모델 정보 조회"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩되지 않았습니다.")
    
    # 실제 모델 정보 반환
    model_details = yolo_model.get_model_info()
    
    return {
        "model_name": "YOLOv7 Press Defect Detection",
        "model_version": "1.0.0",
        "model_file": "press_hole_yolov7_best.pt",
        "huggingface_repo": "23smartfactory/press-defect-detection-model",
        "categories": [
            {"id": cat_id, "name": cat_name} 
            for cat_id, cat_name in model_details['categories'].items()
        ],
        "adaptive_thresholds": model_details['adaptive_thresholds'],
        "model_details": model_details,
        "timestamp": datetime.now().isoformat()
    }

# 임시 예측 엔드포인트 (다음 단계에서 실제 구현)
@app.post("/predict")
async def predict():
    """AI 모델 예측 (임시)"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩되지 않았습니다.")
    
    return {
        "message": "예측 기능은 다음 단계에서 구현됩니다.",
        "status": "placeholder",
        "model_ready": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/file")
async def predict_file():
    """파일 업로드를 통한 예측 (임시)"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩되지 않았습니다.")
    
    return {
        "message": "파일 업로드 예측 기능은 다음 단계에서 구현됩니다.",
        "status": "placeholder",
        "model_ready": True,
        "timestamp": datetime.now().isoformat()
    }