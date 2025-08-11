from fastapi import FastAPI
from contextlib import asynccontextmanager
import os

from app.config.settings import settings
from app.utils.logger import system_log


@asynccontextmanager
async def lifespan(app: FastAPI):

    system_log.info(f"애플리케이션 시작 준비 완료.")
    yield
    system_log.info(f"애플리케이션이 종료됩니다.")


app = FastAPI(title="유압 펌프 고장 예측 시뮬레이터", description="", lifespan=lifespan)
