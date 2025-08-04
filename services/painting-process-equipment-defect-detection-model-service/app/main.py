from contextlib import asynccontextmanager
from fastapi import FastAPI
from .dependencies import load_resources
from .routers import health, info, predict, model_info

# lifespan 이벤트를 처리할 비동기 컨텍스트 매니저를 정의합니다.
# 이 함수는 애플리케이션 시작과 종료 시 리소스를 관리하는 데 사용됩니다.
@asynccontextmanager
async def lifespan_event(app: FastAPI):
    """
    애플리케이션의 생명 주기(lifespan) 동안 리소스를 로드하고 정리합니다.
    - 시작 시: load_resources()를 호출하여 AI 모델, 설정 파일 등을 로드합니다.
    - 종료 시: 필요한 경우 리소스를 정리하는 코드를 추가할 수 있습니다.
    """
    # 애플리케이션 시작 시 실행되는 로직
    print("애플리케이션 시작: AI 모델 및 설정 파일 로딩을 시작합니다.")
    await load_resources()
    print("애플리케이션 시작: 모든 리소스 로딩이 완료되었습니다.")
    
    # yield를 통해 애플리케이션이 실행 상태로 진입합니다.
    yield
    
    # 애플리케이션 종료 시 실행되는 로직
    print("애플리케이션 종료: 리소스를 정리합니다.")

# FastAPI 애플리케이션 인스턴스 생성
# lifespan 인자를 통해 위에서 정의한 lifespan_event 함수를 등록합니다.
app = FastAPI(
    title="E-Coating Issue Prediction Service",
    version="1.0.0",
    description="API for predicting e-coating issues and providing model insights.",
    lifespan=lifespan_event
)

# 라우터 등록
# 라우터는 애플리케이션의 시작/종료 이벤트와는 독립적으로 등록됩니다.
app.include_router(info.router)
app.include_router(health.router)
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(model_info.router, prefix="/model", tags=["Model Info"])
