# E-Coating Issue Prediction Service

## 1. 서비스 소개
이 서비스는 E-coating 공정에서 발생하는 데이터를 분석하여 잠재적인 문제를 예측하고, 문제 발생 시 원인을 파악하여 로그를 생성하는 FastAPI 기반의 AI 서비스입니다.

## 2. 프로젝트 구조 

```text
painting-process-equipment-defect-detection-model-service/
├── app/
│   ├── main.py          # FastAPI 애플리케이션의 메인 진입점 및 라우터 등록
│   ├── dependencies.py  # 공통 의존성 주입 함수 (모델, 설정, explainer 로드 및 제공)
│   ├── models/
│   │   ├── Ecoating_model.pkl   # 학습된 AI 모델 파일 (Joblib 저장)
│   │   └── model_config.yaml    # 모델 관련 설정 및 파라미터 정의
│   ├── routers/
│   │   ├── health.py            # 서비스 헬스 체크 API 엔드포인트
│   │   ├── info.py              # 서비스 기본 정보 API 엔드포인트
│   │   ├── model_info.py        # AI 모델 정보 조회 API 엔드포인트
│   │   └── predict.py           # AI 모델 예측 관련 API 엔드포인트
│   └── services/
│       ├── inference.py         # AI 모델 추론 및 분석 로직 구현
│       └── utils.py             # 공통 유틸리티 함수 및 Pydantic 모델 정의 (로그 저장, 입력 유효성 검사 모델)
├── logs/
│   └── issue_logs.jsonl # 서비스에서 생성된 예측 이슈 로그가 저장되는 파일 (.gitignore에 추가 예정)
├── Dockerfile           # Docker 이미지 빌드 및 컨테이너화 설정
├── requirements.txt     # 프로젝트에 필요한 Python 라이브러리 의존성 목록
└── README.md            # 서비스 개요, 설치/실행 방법, API 엔드포인트 등 문서
```

## 3. 설치 및 실행 방법

### 3.1. 로컬 환경에서 실행
1.  **Python 환경 설정**: Python 3.9 이상 버전을 권장합니다.
    ```bash
    python -m venv venv
    # source venv/bin/activate # Linux/macOS
    source venv/Scripts/activate # Windows
    ```
2.  **의존성 설치**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **애플리케이션 실행**:
    ```bash
    uvicorn app.main:app --reload --port 8001
    uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
    ```
    (개발 중에는 `--reload` 옵션을 사용하여 코드 변경 시 자동 재시작)

### 3.2. Docker를 이용한 실행
1.  **Docker 이미지 빌드**: 프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다.
    ```bash
    docker build -t painting-process-equipment-defect-detection-model-service .
    ```
2.  **Docker 컨테이너 실행**:
    ```bash
    docker run -d -p 8001:8001 painting-process-equipment-defect-detection-model-service
    ```
    `-d` 옵션은 백그라운드에서 실행을 의미하며, `-p 8001:8001`은 호스트의 8001번 포트를 컨테이너의 8001번 포트와 연결합니다.
3.  **Docker 컨테이너 재실행**:
    ```bash
    docker start -a <컨테이너 ID>
    '-a' 옵션을 넣지 않으면 백그라운드에서 실행됩니다.
    ```
## 4. API 엔드포인트

서비스가 시작되면 다음 URL로 API 문서(Swagger UI)에 접근할 수 있습니다: `http://localhost:8001/docs`

| HTTP Method | Endpoint           | Description                                |
| :---------- | :----------------- | :----------------------------------------- |
| `GET`       | `/`                | 서비스의 기본 정보 (이름, 버전, 설명)를 반환합니다. |
| `GET`       | `/health`          | 서비스의 헬스 상태를 확인합니다.             |
| `POST`      | `/predict`         | JSON 데이터를 받아 AI 모델로 예측을 수행하고 로그를 생성합니다. |
| `POST`      | `/predict/file`    | CSV 파일을 업로드하여 AI 모델로 예측을 수행하고 로그를 생성합니다. |
| `GET`       | `/model/info`      | 로드된 AI 모델의 설정 정보를 반환합니다.       |

## 5. 개발 시 주의사항

* **공통 라이브러리 활용**: `app/services/shared/` 디렉토리에 공통으로 사용될 코드를 추가하고 활용하세요. (현재는 `app/services`에 `utils.py`가 포함됨)

---

**Request Body:**

```json
{
  "machineId": "MCH-001",
  "timeStamp": "2025-07-23T15:32:01",
  "thick": 19.9,
  "voltage": 263.0,
  "current": 72.0,
  "temper": 34.0
}
```

Response:

200 OK: 분석 결과 로그 (이상 감지 시)
```json
{
  "machineId": "MCH-001",
  "timeStamp": "2025-07-23T15:32:01",
  "thick": 19.9,
  "voltage": 263.0,
  "current": 72.0,
  "temper": 34.0,
  "issue": "PAINT-EQ-VOL-HIGH-2025-07-23T15:32:01",
  "isSolved": false
}
```
204 No Content: 이상 감지되지 않은 경우 (오차 임계값 이내)
422 Unprocessable Entity: 요청 본문 데이터 형식이 올바르지 않은 경우
500 Internal Server Error: 서버 내부 오류 (설정 로드 실패, 모델 로드 실패 등)

* 로그 파일은 model_config.yaml에 설정된 logs.file_path에 저장됩니다.