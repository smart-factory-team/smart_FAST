from fastapi import FastAPI
from .dependencies import load_resources
# routers 디렉토리의 모듈 임포트
from .routers import health, info, predict, model_info

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="E-Coating Issue Prediction Service",
    version="1.0.0",
    description="API for predicting e-coating issues and providing model insights."
)

# 애플리케이션 시작 시 리소스 로드 (dependencies.py의 함수 호출)
@app.on_event("startup")
async def startup_event():
    await load_resources()

# 라우터 등록
app.include_router(info.router)
app.include_router(health.router)
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(model_info.router, prefix="/model", tags=["Model Info"])