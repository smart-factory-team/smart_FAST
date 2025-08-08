from fastapi import FastAPI
from contextlib import asynccontextmanager
import os

from app.config.settings import settings

app = FastAPI(
    title="유압 펌프 고장 예측 시뮬레이터",
    description=""
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 로그 디렉토리 생성
    if not os.path.exists(settings.LOG_DIR):
        os.makedirs(settings.LOG_DIR, exist_ok=True)