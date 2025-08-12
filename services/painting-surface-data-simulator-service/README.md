# 도장 표면 결함 탐지 데이터 시뮬레이터 서비스

도장 표면의 결함을 탐지하기 위한 실시간 데이터 시뮬레이터 서비스입니다. Azure Blob Storage에서 도장 표면 이미지 데이터를 주기적으로 수집하여 결함 탐지 모델 서비스에 전송하고, 결과를 체계적으로 로깅합니다.

## 🎯 주요 기능

- **🔄 자동화된 데이터 시뮬레이션**: 설정 가능한 간격으로 도장 표면 이미지 데이터 수집 및 모델 추론
- **☁️ Azure Blob Storage 연동**: 클라우드 기반 이미지 데이터 관리 및 실시간 접근
- **🤖 모델 서비스 통신**: 도장 표면 결함 탐지 모델과의 HTTP 통신 및 예측 요청
- **📊 실시간 모니터링**: 결함 탐지 결과, 시스템 상태, 연결 상태 실시간 모니터링
- **📝 체계적 로깅**: 결함 탐지, 정상 처리, 오류 상황을 JSON 형태로 체계적 기록
- **⚡ 비동기 처리**: FastAPI 기반의 고성능 비동기 처리로 효율적인 데이터 시뮬레이션

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Azure Blob    │    │   Simulator      │    │   Painting      │
│   Storage       │◄──►│   Service        │◄──►│   Surface       │
│   (Images)      │    │   (Port 8012)    │    │   Model         │
│                 │    │                  │    │   (Port 8002)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │   Logger         │
                        │   (JSON Files)   │
                        │   - logs/        │
                        │   - errors/      │
                        └──────────────────┘
```

## 🚀 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

**핵심 의존성 패키지:**
- **FastAPI & Uvicorn**: 웹 서비스 프레임워크
- **Azure Storage Blob**: 클라우드 스토리지 연동
- **APScheduler**: 주기적 작업 스케줄링
- **httpx**: 모델 서비스와의 HTTP 통신
- **pydantic-settings**: 환경 변수 관리

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 설정하세요:

```env
# Azure Storage 설정 (필수)
AZURE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;EndpointSuffix=core.windows.net
AZURE_CONTAINER_NAME=simulator-data
PAINTING_DATA_FOLDER=painting-surface

# 스케줄러 설정
SCHEDULER_INTERVAL_MINUTES=1
BATCH_SIZE=10

# 모델 서비스 설정
PAINTING_MODEL_URL=http://painting-model-service:8002

# 로깅 설정
LOG_DIRECTORY=logs
LOG_FILENAME=painting_defect_detections.json
ERROR_LOG_FILENAME=painting_errors.json

# HTTP 설정
HTTP_TIMEOUT=30
MAX_RETRIES=3
```

### 3. 서비스 실행

```bash
# 개발 모드
uvicorn app.main:app --host 0.0.0.0 --port 8012 --reload

# 프로덕션 모드
uvicorn app.main:app --host 0.0.0.0 --port 8012
```

**포트 정보:**
- **시뮬레이터 서비스**: 포트 8012
- **도장 표면 모델 서비스**: 포트 8002 (외부 연결)

## 📡 API 엔드포인트

### 🏠 기본 정보

#### 서비스 정보 조회
```http
GET /
```
**응답 예시:**
```json
{
  "service": "Painting Surface Data Simulator Service",
  "version": "1.0.0",
  "status": "running",
  "target_model": "painting-surface-defect-detection",
  "scheduler_status": {
    "is_running": true,
    "scheduler_interval_minutes": 1,
    "batch_size": 10
  },
  "azure_storage": {
    "container": "simulator-data",
    "data_folder": "painting-surface",
    "connection_status": "connected"
  }
}
```

#### 헬스 체크
```http
GET /health
```
**응답:**
```json
{
  "status": "healthy"
}
```

### 🎮 시뮬레이터 제어

#### 시뮬레이터 시작
```http
POST /simulator/start
```
**응답 예시:**
```json
{
  "message": "시뮬레이터가 시작되었습니다.",
  "status": {
    "is_running": true,
    "scheduler_interval_minutes": 1,
    "batch_size": 10
  }
}
```

#### 시뮬레이터 중지
```http
POST /simulator/stop
```
**응답 예시:**
```json
{
  "message": "시뮬레이터가 중지되었습니다.",
  "status": {
    "is_running": false,
    "scheduler_interval_minutes": 1,
    "batch_size": 10
  }
}
```

#### 시뮬레이터 상태 조회
```http
GET /simulator/status
```
**응답 예시:**
```json
{
  "is_running": true,
  "scheduler_interval_minutes": 1,
  "batch_size": 10,
  "painting_surface_service_health": true
}
```

### 📊 로그 관리

#### 최근 로그 조회
```http
GET /simulator/logs/recent
```
**응답 예시:**
```json
{
  "logs": [
    {
      "timestamp": "2024-01-01T12:00:00",
      "service_name": "painting-surface",
      "prediction": {
        "status": "anomaly",
        "defect_count": 2,
        "total_count": 5,
        "defect_ratio": 0.4,
        "combined_logic": "총 5개 이미지 중 2개에서 결함 탐지 → 최종: anomaly"
      },
      "original_data": {
        "images": ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]
      }
    }
  ],
  "total_count": 1
}
```

### 🔗 연결 테스트

#### Azure Storage 연결 테스트
```http
POST /test/azure-storage-connection
```
**성공 응답:**
```json
{
  "status": "success",
  "message": "Azure Storage 연결 성공",
  "file_count": 15,
  "sample_files": [
    "painting-surface/image1.jpg",
    "painting-surface/image2.jpg",
    "painting-surface/image3.jpg",
    "painting-surface/image4.jpg",
    "painting-surface/image5.jpg"
  ]
}
```

**실패 응답:**
```json
{
  "status": "error",
  "message": "Azure Storage 연결 실패: 연결 문자열이 올바르지 않습니다."
}
```

#### 모델 서비스 연결 테스트
```http
POST /test/models-connection
```
**성공 응답:**
```json
{
  "status": "success",
  "service_name": "painting-surface-defect-detection",
  "healthy": true,
  "message": "도장 표면 결함탐지 서비스 연결 성공"
}
```

**실패 응답:**
```json
{
  "status": "error",
  "service_name": "painting-surface-defect-detection",
  "healthy": false,
  "message": "도장 표면 결함탐지 서비스 연결 실패"
}
```

## ⚙️ 시뮬레이션 프로세스

### 1. **초기화 단계**
- Azure Storage 연결 및 인증
- 도장 표면 결함 탐지 모델 서비스 헬스 체크
- 로그 디렉토리 생성 및 설정

### 2. **데이터 수집 단계**
- Azure Blob Storage에서 `painting-surface/` 폴더 내 이미지 파일 목록 조회
- 지원 형식: `.jpg`, `.jpeg`, `.png`, `.bmp`
- 순차적 이미지 인덱싱으로 데이터 시뮬레이션

### 3. **모델 추론 단계**
- 각 이미지를 Azure에서 다운로드
- 파일 업로드 방식으로 모델 서비스에 전송
- 결함 탐지 결과 수신 및 분석

### 4. **결과 처리 단계**
- 결함 탐지 여부에 따른 결과 분류
- 상세 정보 로깅 (결함 개수, 총 이미지 수, 비율 등)
- 정상/이상 상태 판정

### 5. **로깅 및 모니터링**
- JSON 형태로 구조화된 로그 저장
- 실시간 콘솔 출력
- 에러 상황 별도 로그 파일 관리

## 📋 결함 탐지 결과 구조

### 이미지별 결과
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "service_name": "painting-surface",
  "prediction": {
    "status": "anomaly",
    "defect_count": 2,
    "total_count": 5,
    "defect_ratio": 0.4,
    "combined_logic": "총 5개 이미지 중 2개에서 결함 탐지 → 최종: anomaly"
  },
  "original_data": {...}
}
```

### 결합 결과 로직
- **정상 (normal)**: 모든 이미지에서 결함 미탐지
- **이상 (anomaly)**: 하나 이상의 이미지에서 결함 탐지
- **결함 비율**: `defect_count / total_count`

## 🐳 Docker 실행

### 이미지 빌드
```bash
docker build -t painting-surface-data-simulator-service .
```

### 컨테이너 실행
```bash
docker run -d -p 8012:8012 \
  -e AZURE_CONNECTION_STRING="your_connection_string" \
  -e AZURE_CONTAINER_NAME="simulator-data" \
  -e PAINTING_MODEL_URL="http://host.docker.internal:8002" \
  --name painting-data-simulator \
  painting-surface-data-simulator-service
```

## 🔍 모니터링 및 디버깅

### 시뮬레이터 상태 확인
```bash
# 상태 조회
curl http://localhost:8012/simulator/status

# 헬스 체크
curl http://localhost:8012/health

# 최근 로그
curl http://localhost:8012/simulator/logs/recent
```

### 로그 파일 구조
```
logs/
├── painting_defect_detections.json    # 결함 탐지 결과 로그
└── painting_errors.json               # 에러 로그
```

### 연결 상태 확인
```bash
# Azure Storage 연결 테스트
curl -X POST http://localhost:8012/test/azure-storage-connection

# 모델 서비스 연결 테스트
curl -X POST http://localhost:8012/test/models-connection
```

## 🧪 개발 및 테스트

### 테스트 실행
```bash
pytest tests/
```

### 코드 품질 관리
```bash
# 코드 포맷팅
black app/
isort app/

# 린팅
flake8 app/
mypy app/
```

## ⚠️ 주의사항

1. **Azure Storage 연결**: `AZURE_CONNECTION_STRING` 환경 변수는 필수입니다.
2. **모델 서비스**: 도장 표면 결함 탐지 모델 서비스가 실행 중이어야 합니다.
3. **포트 충돌**: 포트 8012가 사용 가능한지 확인하세요.
4. **로그 디스크 공간**: 로그 파일이 지속적으로 증가하므로 디스크 공간을 모니터링하세요.

## 🔧 설정 옵션

### 스케줄러 설정
- **`SCHEDULER_INTERVAL_MINUTES`**: 데이터 수집 간격 (기본값: 1분)
- **`BATCH_SIZE`**: 한 번에 처리할 이미지 수 (기본값: 10개)

### HTTP 설정
- **`HTTP_TIMEOUT`**: 모델 서비스 요청 타임아웃 (기본값: 30초)
- **`MAX_RETRIES`**: 재시도 최대 횟수 (기본값: 3회)

### 로깅 설정
- **`LOG_DIRECTORY`**: 로그 저장 디렉토리 (기본값: `logs`)
- **`LOG_FILENAME`**: 결함 탐지 로그 파일명
- **`ERROR_LOG_FILENAME`**: 에러 로그 파일명

## 📞 지원 및 문의

서비스 관련 문제나 개선 사항이 있으시면 개발팀에 문의해주세요.

---

**버전**: 1.0.0  
**최종 업데이트**: 2024년 1월  
**라이선스**: 내부 사용 전용
