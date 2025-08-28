FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# 비루트 유저 생성 (uid/gid 고정: 1000)
RUN useradd -u 1000 -m appuser

# 먼저 requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY app/ ./app/

# 앱이 쓸 디렉토리들 생성
RUN mkdir -p \
      /app/models \
      /app/uploads \
      /app/temp \
      /app/logs \
      /app/model_cache \
      /app/cache \
    && chown -R appuser:appuser /app

# 비루트로 실행
USER appuser

EXPOSE 8014
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8014"]
