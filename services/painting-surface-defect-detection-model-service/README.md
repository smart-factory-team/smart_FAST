### CI/CD 테스트 

# 도장 표면 결함 탐지 모델 서비스

자동차 제조 공정에서 도장 표면의 결함을 탐지하는 AI 모델 서비스입니다.

## 🎯 기능

- **4가지 결함 탐지**: 먼지/오염, 흘러내림, 스크래치, 물 자국
- **Hugging Face 모델**: [23smartfactory/painting-surface-defect](https://huggingface.co/23smartfactory/painting-surface-defect)에서 YOLOv8 모델 로드
- **다양한 입력 방식**: 파일 업로드, Base64 인코딩 지원
- **실시간 추론**: FastAPI 기반 고성능 API 서비스
- **Docker 컨테이너화**: 완전한 컨테이너 환경 지원
- **포괄적인 테스트**: 단위 테스트, 통합 테스트 완비
- **정상/결함 구분**: 신뢰도 기반 자동 판정 시스템

## 🏗️ 아키텍처

```
app/
├── main.py              # FastAPI 애플리케이션 (lifespan 관리)
├── models/
│   └── yolo_model.py    # Hugging Face 모델 로더
├── routers/
│   └── predict.py       # API 엔드포인트 (파일 업로드 + Base64)
└── services/
    └── inference.py     # 추론 서비스 (비동기 처리)
tests/
├── test_main.py         # FastAPI 메인 애플리케이션 테스트
├── test_predict_router.py # 예측 라우터 테스트
└── test_model_loader.py # 모델 로더 테스트
```

## 🚀 시작하기

### 1. 가상환경 설정

```bash
# 가상환경 생성 (Python 3.10 권장)
python3.10 -m venv venv

# 가상환경 활성화
source venv/Scripts/activate  # Windows (Git Bash)
# source venv/bin/activate    # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 서비스 실행

```bash
# 개발 모드
uvicorn app.main:app --reload --port 8002

# 또는 스크립트 사용
./run_dev.bat    # Windows
./run_dev.sh     # Linux/Mac

# 또는 Docker
docker build -t painting-surface-defect-detection .
docker run -p 8002:8002 painting-surface-defect-detection
```

### 3. Docker Compose (전체 서비스)

```bash
# 프로젝트 루트에서
cd ../../infrastructure
docker-compose up --build
```

## 📡 API 엔드포인트

### 기본 엔드포인트

- `GET /health` - 헬스 체크
- `GET /ready` - 모델 로딩 상태
- `GET /startup` - 서비스 시작 준비 상태

### 예측 엔드포인트

- `POST /api/predict` - 파일 업로드 기반 결함 탐지
- `POST /api/predict/base64` - Base64 이미지 기반 결함 탐지
- `GET /api/model/info` - 모델 정보 조회

## 🔧 사용 예시

### 파일 업로드 예측

```bash
curl -X POST "http://localhost:8002/api/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@testdata.jpg" \
  -F "confidence_threshold=0.5"
```

### Base64 이미지 예측

```bash
curl -X POST "http://localhost:8002/api/predict/base64" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<base64_encoded_image>",
    "confidence_threshold": 0.5
  }'
```

### 모델 정보 조회

```bash
curl -X GET "http://localhost:8002/api/model/info" \
  -H "accept: application/json"
```

## 📊 응답 형식

### 정상 이미지 응답
```json
{
  "predictions": [],
  "image_shape": [640, 640, 3],
  "confidence_threshold": 0.5,
  "timestamp": "2024-01-01T12:00:00",
  "model_source": "Hugging Face"
}
```

### 결함 이미지 응답
```json
{
  "predictions": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "dirt",
      "area": 1500.0
    }
  ],
  "image_shape": [640, 640, 3],
  "confidence_threshold": 0.5,
  "timestamp": "2024-01-01T12:00:00",
  "model_source": "Hugging Face"
}
```

## 🎨 탐지 가능한 결함

| 클래스 ID | 클래스명 | 설명 | 탐지 특징 |
|-----------|----------|------|-----------|
| 0 | dirt | 먼지/오염 | 작은 점 형태, 낮은 신뢰도에서도 탐지 |
| 1 | runs | 흘러내림 | 세로 방향 흐름, 중간 신뢰도 |
| 2 | scratch | 스크래치 | 선형 결함, 높은 신뢰도 |
| 3 | water_marks | 물 자국 | 넓은 영역, 다양한 신뢰도 |

## ⚙️ 신뢰도 임계값 설정

### 임계값별 특징

| 임계값 | 탐지 특성 | 용도 | 권장 상황 |
|--------|-----------|------|-----------|
| **0.1** | 매우 민감, 노이즈 많음 | 초기 탐색 | 모든 가능한 결함 확인 |
| **0.3** | 민감, 중간 노이즈 | 일반 검사 | 균형잡힌 탐지 |
| **0.5** | 균형잡힌 성능 | **기본값** | **일반적인 운영** |
| **0.7** | 엄격한 기준, 높은 정확도 | 품질 관리 | 확실한 결함만 탐지 |

### 임계값 설정 방법

**1. API 호출 시 동적 설정**
```bash
# 높은 정확도로 검사
curl -X POST "http://localhost:8002/api/predict" \
  -F "image=@image.jpg" \
  -F "confidence_threshold=0.7"

# 민감하게 검사
curl -X POST "http://localhost:8002/api/predict" \
  -F "image=@image.jpg" \
  -F "confidence_threshold=0.3"
```

**2. 환경 변수로 설정**
```bash
export CONFIDENCE_THRESHOLD=0.7
uvicorn app.main:app --reload --port 8002
```

**3. 코드에서 기본값 변경**
```python
# app/routers/predict.py
confidence_threshold: float = Form(0.7, ge=0.0, le=1.0)
```

## 🔍 정상/결함 구분 방법

### 판정 기준

**정상 이미지:**
- `predictions` 배열이 비어있음 (`[]`)
- 어떤 임계값에서도 결함이 탐지되지 않음

**결함 이미지:**
- `predictions` 배열에 결함 정보 포함
- 신뢰도에 따라 결함의 확실성 판단

### 판정 로직 예시

```python
def classify_image(response):
    """이미지 분류 함수"""
    if not response["predictions"]:
        return "NORMAL"
    
    # 높은 신뢰도 결함 확인
    high_conf_defects = [
        p for p in response["predictions"] 
        if p["confidence"] > 0.5
    ]
    
    if high_conf_defects:
        defect_types = [p["class_name"] for p in high_conf_defects]
        return f"DEFECT: {', '.join(defect_types)}"
    else:
        return "NORMAL"
```

### 실제 테스트 결과

**정상 이미지 (테라코타 색상 표면):**
```json
{
  "predictions": [],
  "image_shape": [897, 492, 3],
  "confidence_threshold": 0.5,
  "결과": "정상"
}
```

**결함 이미지 (스크래치 + 먼지):**
```json
{
  "predictions": [
    {
      "bbox": [930.7, 85.2, 1024.0, 518.0],
      "confidence": 0.659,
      "class_id": 2,
      "class_name": "scratch",
      "area": 40379.34
    },
    {
      "bbox": [590.0, 13.5, 599.1, 26.9],
      "confidence": 0.578,
      "class_id": 0,
      "class_name": "dirt",
      "area": 122.77
    }
  ],
  "image_shape": [768, 1024, 3],
  "confidence_threshold": 0.5,
  "결과": "결함"
}
```

**임계값별 비교 결과:**
```json
// 임계값 0.1 (매우 민감)
{
  "predictions": [36개 결함 탐지],
  "confidence_threshold": 0.1
}

// 임계값 0.3 (일반적)
{
  "predictions": [9개 결함 탐지],
  "confidence_threshold": 0.3
}

// 임계값 0.5 (기본값)
{
  "predictions": [4개 결함 탐지],
  "confidence_threshold": 0.5
}

// 임계값 0.6 (엄격)
{
  "predictions": [1개 결함 탐지],
  "confidence_threshold": 0.6
}
```

## 🔧 환경 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `HUGGING_FACE_ORG` | 23smartfactory | Hugging Face 조직명 |
| `HUGGING_FACE_REPO` | painting-surface-defect | 모델 저장소명 |
| `HUGGING_FACE_MODEL_NAME` | 23smartfactory/painting-surface-defect | 전체 모델 경로 |
| `CONFIDENCE_THRESHOLD` | 0.5 | 기본 신뢰도 임계값 |
| `PORT` | 8002 | 서비스 포트 |
| `HOST` | 0.0.0.0 | 서비스 호스트 |

## 📝 개발 가이드

### 모델 업데이트

1. Hugging Face에 새 모델 업로드
2. `models/yolo_model.py`에서 모델 파일명 업데이트
3. 서비스 재시작

### 로컬 개발

```bash
# 가상환경 생성 (Python 3.10 권장)
python3.10 -m venv venv
source venv/Scripts/activate  # Windows (Git Bash)
# source venv/bin/activate    # Linux/Mac

# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
uvicorn app.main:app --reload --port 8002
```

### 코드 품질

- **불필요한 코드 제거**: 사용하지 않는 함수 및 설정 정리
- **단순한 임계값 설정**: 복잡한 결함별 임계값 대신 단일 임계값 사용
- **Docker 최적화**: 시스템 라이브러리 포함으로 안정성 확보
- **OpenCV GUI 비활성화**: 서버 환경에서 GUI 관련 오류 방지

## 🧪 테스트

### 테스트 실행

```bash
# 모든 테스트 실행
pytest tests/ -v

# 특정 테스트만 실행
pytest tests/test_main.py -v
pytest tests/test_predict_router.py -v
pytest tests/test_model_loader.py -v

# 개별 테스트 파일 실행
python tests/test_main.py
python tests/test_predict_router.py
python tests/test_model_loader.py

# 또는 스크립트 사용
./run_tests.bat    # Windows
./run_tests.sh     # Linux/Mac
```

### 테스트 구조

```
tests/
├── test_main.py              # FastAPI 메인 애플리케이션 테스트
│   ├── 앱 인스턴스 생성 및 설정
│   ├── 라우터 등록 및 엔드포인트 설정
│   ├── OpenAPI 스키마 생성
│   ├── 문서화 엔드포인트 접근성
│   ├── 상태 체크 엔드포인트
│   ├── 요청 검증 및 오류 처리
│   ├── 성능 및 안정성 테스트
│   └── 엣지 케이스 처리
├── test_predict_router.py    # 예측 라우터 테스트
│   ├── 파일 업로드 방식 엔드포인트
│   ├── Base64 방식 엔드포인트
│   ├── 입력 검증 (이미지, 신뢰도 임계값)
│   ├── 응답 구조 검증
│   ├── 오류 처리
│   ├── Pydantic 모델 검증
│   └── 모델 정보 엔드포인트
└── test_model_loader.py      # 모델 로더 테스트
    ├── Hugging Face 모델 로딩
    ├── 설정 파일 처리
    ├── 클래스 매핑
    ├── 임계값 설정
    ├── 모델 유효성 검사
    ├── 환경 변수 오버라이드
    └── 모델 정보 조회
```

### 테스트 커버리지

- **FastAPI 애플리케이션**: 앱 생성, 라우터 등록, 엔드포인트 접근성, lifespan 관리
- **예측 라우터**: API 요청/응답 검증, 입력 유효성 검사, 에러 처리, 파일 업로드/Base64 방식
- **모델 로더**: Hugging Face 모델 로딩, 설정 파일 처리, 환경 변수, 모델 검증
- **통합 테스트**: 실제 API 호출, 동시 요청 처리, 성능 테스트

### 테스트 특징

- **Mock 활용**: 외부 의존성(Hugging Face, YOLO 모델) 적절히 모킹
- **엣지 케이스 처리**: 오류 상황과 경계값 테스트 포함
- **한국어 주석**: 테스트 목적과 기능을 명확히 설명
- **실용적인 테스트**: 실제 사용 시나리오를 반영한 테스트 케이스

## 🔍 모니터링

- **API 문서**: http://localhost:8002/docs
- **ReDoc 문서**: http://localhost:8002/redoc
- **헬스 체크**: http://localhost:8002/health
- **모델 정보**: http://localhost:8002/api/model/info
- **서비스 상태**: http://localhost:8002/ready

## 🚨 주의사항

### 파일 업로드 제한
- **파일 크기**: 최대 10MB
- **지원 형식**: 이미지 파일만 (image/*)
- **권장 형식**: JPG, PNG

### Base64 인코딩
- **최소 길이**: 100자 이상
- **유효성 검사**: 올바른 Base64 인코딩 필수
- **이미지 형식**: JPG, PNG 지원

### 신뢰도 임계값
- **범위**: 0.0 ~ 1.0
- **기본값**: 0.5
- **검증**: Pydantic Field 검증 적용

### 성능 고려사항
- **모델 로딩**: 서비스 시작 시 약 30초 소요
- **추론 속도**: 이미지당 약 1-3초 (GPU 사용 시 더 빠름)
- **메모리 사용량**: 약 2-4GB (모델 크기에 따라 다름)
