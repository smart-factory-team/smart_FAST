FROM python:3.10

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# config.py에서 필요한 모든 디렉토리들을 미리 생성
RUN mkdir -p \
    /app/models \
    /app/uploads \
    /app/temp \
    /app/logs \
    /app/model_cache \
    /app/cache \
    && chmod -R 777 /app

# HuggingFace 캐시 디렉토리 설정
ENV HF_HUB_CACHE=/app/cache

# 애플리케이션 코드 복사
COPY app/ ./app/

EXPOSE 8005
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]