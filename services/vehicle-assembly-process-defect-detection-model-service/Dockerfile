FROM python:3.10

WORKDIR /app

# 시스템 의존성 설치 (opencv-python을 위한 최소 패키지)
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r  \
    requirements.txt

# HuggingFace 캐시 디렉토리 생성
RUN mkdir -p /app/cache && chmod 755 /app/cache
ENV HF_HUB_CACHE=/app/cache

# 애플리케이션 코드 복사
COPY app/ ./app/

EXPOSE 8005
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]