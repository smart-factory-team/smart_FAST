"""
FastAPI 라우터 모듈

모든 API 라우터들을 중앙에서 관리합니다.
각 라우터는 특정 기능별로 분리되어 있습니다.

사용법:
    from app.routers import root, health, predict, model

    app.include_router(root.router)
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(model.router)
"""

# 각 라우터 모듈 import
from . import root
from . import health
from . import predict
from . import model

# 편의를 위한 라우터 객체 직접 export
from .root import router as root_router
from .health import router as health_router
from .predict import router as predict_router
from .model import router as model_router

# 모든 라우터를 리스트로 관리 (main.py에서 일괄 등록용)
ALL_ROUTERS = [
    root_router,
    health_router,
    predict_router,
    model_router
]

# 라우터별 정보
ROUTER_INFO = {
    "root": {
        "router": root_router,
        "prefix": "",
        "tags": ["Root"],
        "description": "서비스 기본 정보"
    },
    "health": {
        "router": health_router,
        "prefix": "",
        "tags": ["Health"],
        "description": "헬스체크 및 상태 모니터링"
    },
    "predict": {
        "router": predict_router,
        "prefix": "/predict",
        "tags": ["Prediction"],
        "description": "AI 모델 예측 서비스"
    },
    "model": {
        "router": model_router,
        "prefix": "/model",
        "tags": ["Model"],
        "description": "모델 정보 및 관리"
    }
}

# 라우터 등록 헬퍼 함수
def register_all_routers(app):
    """모든 라우터를 FastAPI 앱에 등록

    Args:
        app: FastAPI 인스턴스
    """
    for router_name, info in ROUTER_INFO.items():
        app.include_router(
            info["router"],
            prefix=info["prefix"],
            tags=info["tags"]
        )

# 특정 라우터만 등록하는 함수
def register_router(app, router_name: str):
    """특정 라우터만 등록

    Args:
        app: FastAPI 인스턴스
        router_name: 등록할 라우터 이름
    """
    if router_name in ROUTER_INFO:
        info = ROUTER_INFO[router_name]
        app.include_router(
            info["router"],
            prefix=info["prefix"],
            tags=info["tags"]
        )
    else:
        available = ", ".join(ROUTER_INFO.keys())
        raise ValueError(f"Unknown router: {router_name}. Available: {available}")

# 라우터 정보 조회
def get_router_info(router_name: str = None):
    """라우터 정보 조회

    Args:
        router_name: 조회할 라우터 이름 (None이면 전체)

    Returns:
        라우터 정보 딕셔너리
    """
    if router_name:
        return ROUTER_INFO.get(router_name)
    return ROUTER_INFO

# 엔드포인트 목록 조회
def get_all_endpoints():
    """모든 라우터의 엔드포인트 목록 반환"""
    endpoints = []
    for router_name, info in ROUTER_INFO.items():
        router = info["router"]
        prefix = info["prefix"]

        for route in router.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                for method in route.methods:
                    if method != 'HEAD':  # HEAD 메서드 제외
                        endpoint = {
                            "path": prefix + route.path,
                            "method": method,
                            "name": route.name,
                            "router": router_name,
                            "tags": info["tags"]
                        }
                        endpoints.append(endpoint)

    return endpoints

# 모듈 정보
__version__ = "1.0.0"
__description__ = "자동차 의장 불량 탐지 API 라우터"

# 익스포트할 주요 요소들
__all__ = [
    # 라우터 모듈들
    "root",
    "health",
    "predict",
    "model",

    # 라우터 객체들
    "root_router",
    "health_router",
    "predict_router",
    "model_router",

    # 라우터 관리
    "ALL_ROUTERS",
    "ROUTER_INFO",
    "register_all_routers",
    "register_router",
    "get_router_info",
    "get_all_endpoints"
]