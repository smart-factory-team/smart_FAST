from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import simulation
from app.utils.logging import setup_logging

# 로깅 설정
setup_logging()

# FastAPI 앱 초기화
app = FastAPI(title="의장 공정 불량 탐지 시뮬레이터", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(simulation.router, tags=["simulation"])

@app.get("/")
async def root():
    return {"message": "의장 공정 불량 탐지 시뮬레이터 API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/ready")
async def health_check():
    return {"status": "ready", "version": "1.0.0"}

@app.get("/startup")
async def health_check():
    return {"status": "startup", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)