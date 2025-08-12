# 도장 표면 결함 탐지 시뮬레이터 서비스

도장 표면의 결함을 탐지하기 위한 데이터 시뮬레이터 서비스입니다. Azure Blob Storage에서 이미지 데이터를 읽어와 모델 서비스에 전송하고 결함 탐지 결과를 로깅합니다.

## 주요 기능

- **Azure Blob Storage 연동**: 클라우드에서 이미지 데이터 관리
- **자동화된 시뮬레이션**: 주기적인 이미지 데이터 수집 및 모델 추론
- **실시간 모니터링**: 결함 탐지 결과 및 시스템 상태 모니터링
- **로깅 시스템**: 결함 탐지, 정상 처리, 오류 상황을 체계적으로 기록
- **파일 업로드 기반 예측**: 이미지를 직접 모델 서비스에 전송하여 결함 감지

## 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Azure Blob    │    │   Simulator      │    │   Model         │
│   Storage       │◄──►│   Service        │◄──►│   Service       │
│   (Images)      │    │   (Port 8012)    │    │   (Port 8002)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │   Logger         │
                        │   (JSON Files)   │
                        └──────────────────┘
```

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

**최적화된 의존성**: 모델 관련 불필요한 패키지 제거 (ultralytics, torch, opencv 등)
- **핵심 서비스**: FastAPI, Uvicorn, Pydantic
- **Azure 연동**: azure-storage-blob, aiohttp
- **스케줄링**: APScheduler
- **HTTP 통신**: httpx (모델 서비스와의 통신용)

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 설정하세요:

```env
# Azure Storage 설정
AZURE_CONNECTION_STRING=your_azure_connection_string
AZURE_CONTAINER_NAME=simulator-data
PAINTING_DATA_FOLDER=painting-surface

# 스케줄러 설정
SCHEDULER_INTERVAL_MINUTES=1
BATCH_SIZE=10

# 모델 서비스 설정
PAINTING_MODEL_URL=http://localhost:8002

# 로깅 설정
LOG_DIRECTORY=logs
LOG_FILENAME=painting_anomaly_detections.json
ERROR_LOG_FILENAME=painting_errors.json

# HTTP 설정
HTTP_TIMEOUT=30
MAX_RETRIES=3
```

### 3. 서비스 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8012 --reload
```

**포트 정보:**
- **시뮬레이터 서비스**: 포트 8012 (자체 실행)
- **연결하는 모델 서비스**: 포트 8002 (외부 서비스)

## API 엔드포인트

### 시뮬레이터 제어

#### 시뮬레이터 시작
```http
POST /simulator/start
```

#### 시뮬레이터 중지
```http
POST /simulator/stop
```

#### 시뮬레이터 상태 조회
```http
GET /simulator/status
```

### 로그 관리

#### 로그 조회
```http
GET /simulator/logs?log_type=all&limit=100
```

**로그 타입:**
- `all`: 모든 로그
- `anomaly`: 이상 감지 로그
- `error`: 오류 로그
- `normal`: 정상 처리 로그

#### 로그 요약 조회
```http
GET /simulator/logs/summary
```

#### 로그 정리
```http
POST /simulator/logs/clear?log_type=all
```

### 시스템 상태

#### 헬스 체크
```http
GET /simulator/health
```

#### 연결 테스트
```http
GET /test/azure
GET /test/model
```

#### Azure 이미지 목록
```http
GET /simulator/azure/images?limit=20
```

## 시뮬레이션 프로세스

1. **초기화**: Azure Storage 연결 및 모델 서비스 헬스 체크
2. **이미지 수집**: Azure Blob Storage에서 도장 표면 이미지 파일 목록 조회
3. **데이터 시뮬레이션**: 순차적으로 이미지 파일을 읽어 Azure에서 다운로드
4. **모델 추론**: 이미지를 파일 업로드 방식으로 모델 서비스에 전송하여 결함 감지
5. **결과 분석**: 결함 감지 여부에 따른 결과 분류 및 상세 정보 로깅
6. **로깅**: 이상 감지, 정상 처리, 오류 상황을 JSON 파일에 기록
7. **주기적 실행**: 설정된 간격으로 위 과정을 반복

## 결함 감지 결과 상세 정보

시뮬레이터는 각 이미지별로 다음과 같은 상세 정보를 로깅합니다:

- **이미지 정보**: 파일명, 크기, 형태
- **결함 탐지 결과**: 탐지된 결함의 종류 (스크래치, 얼룩 등)
- **결함 위치**: 바운딩 박스 좌표 (x, y, width, height)
- **결함 크기**: 면적 (픽셀 단위)
- **신뢰도**: 모델의 예측 신뢰도 (0.0 ~ 1.0)
- **타임스탬프**: 처리 시간
- **모델 소스**: 사용된 모델 정보

## 모니터링 및 디버깅

### 시뮬레이터 상태 모니터링

```bash
# 상태 조회
curl http://localhost:8012/simulator/status

# 헬스 체크
curl http://localhost:8012/simulator/health

# 로그 요약
curl http://localhost:8012/simulator/logs/summary
```

### 로그 파일 위치

```
logs/
├── painting_anomaly_detections.json
├── painting_normal_processing.json
└── painting_errors.json
```

## Docker 실행

### 이미지 빌드
```bash
docker build -t painting-surface-data-simulator-service .
```

### 컨테이너 실행
```bash
# 모델 서비스와 별도로 실행 (포트 8002 사용)
docker run -d -p 8012:8012 \
  -e PAINTING_MODEL_URL=http://host.docker.internal:8002 \
  --name painting-data-simulator \
  painting-surface-data-simulator-service
```

**중요**: `PAINTING_MODEL_URL`을 `http://host.docker.internal:8002`로 설정하여 호스트의 모델 서비스에 연결

## 개발 환경

### 테스트 실행

```bash
pytest tests/
```

### 코드 품질

```bash
# 코드 포맷팅
black app/
isort app/

# 린팅
flake8 app/
mypy app/
```

## 성능 최적화

### 의존성 최적화
- **모델 관련 패키지 제거**: ultralytics, torch, opencv 등 불필요한 의존성 제거
- **경량화**: 시뮬레이터 역할에 필요한 핵심 패키지만 유지
- **빌드 시간 단축**: 불필요한 패키지 다운로드 및 설치 시간 절약

### 메모리 사용량
- **이미지 처리**: Azure에서 이미지를 스트리밍 방식으로 처리
- **배치 처리**: 설정 가능한 배치 크기로 메모리 사용량 제어
