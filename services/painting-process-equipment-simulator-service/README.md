# Painting Process Equipment Simulator Service

## 1. 서비스 소개
이 서비스는 **Painting Process Equipment Defect Detection Model Service**를 테스트하고 모니터링하기 위한 시뮬레이터입니다.

주기적으로 Azure Blob Storage에 저장된 CSV 데이터를 읽어와 실시간 공정 데이터처럼 모델 서비스의 예측 API를 호출합니다. 그 후, 모델 서비스로부터 받은 결과(정상 또는 이상 감지)를 콘솔과 로그 파일에 기록하여 시스템의 동작을 검증하는 역할을 합니다.

## 2. 주요 기능
- **주기적 데이터 시뮬레이션**: `APScheduler`를 사용하여 설정된 시간 간격마다 자동으로 데이터를 생성하고 예측을 요청합니다.
- **Azure Blob Storage 연동**: Azure Blob Storage에 저장된 실제 공정 데이터 기반의 CSV 파일을 읽어 시뮬레이션에 사용합니다.
- **모델 서비스 연동**: `HTTPX` 클라이언트를 사용하여 `painting-process-equipment-defect-detection-model-service`의 API를 비동기적으로 호출합니다.
- **상태 로깅**: 모델의 예측 결과를 `anomaly_logger`를 통해 이상(anomaly) 또는 정상(normal) 상태로 구분하여 로그를 기록합니다.
- **Docker 지원**: Dockerfile을 통해 컨테이너 환경에서 쉽게 서비스를 빌드하고 실행할 수 있으며, Docker Compose를 통한 통합 관리에도 용이합니다.

## 3. 프로젝트 구조

```text
painting-process-equipment-simulator-service/
├── app/
│   ├── main.py                      # FastAPI 애플리케이션의 메인 진입점
│   ├── config/
│   │   └── settings.py              # Pydantic-settings를 이용한 환경 변수 및 설정 관리
│   ├── routers/
│   │   ├── simulator_router.py      # 시뮬레이터 시작/중지/상태 확인 API
│   │   └── test_connection_router.py # 외부 서비스(Azure, 모델) 연결 테스트 API
│   ├── services/
│   │   ├── scheduler_service.py     # APScheduler를 사용한 핵심 스케줄링 로직
│   │   ├── model_client.py          # 모델 예측 서비스 API 호출 클라이언트
│   │   └── azure_storage.py         # Azure Blob Storage 데이터 처리 서비스
│   └── utils/
│       └── logger.py                # 이상 및 정상 로그 기록 유틸리티
├── logs/                            # 시뮬레이션 결과 로그가 저장되는 디렉토리
├── .env                             # Azure 연결 문자열 등 민감한 환경 변수 파일
├── Dockerfile                       # Docker 이미지 빌드 설정
├── requirements.txt                 # Python 라이브러리 의존성 목록
└── README.md                        # 서비스 개요 및 사용법 문서
```

## 4. 설치 및 실행 방법

### 4.1. 로컬 환경에서 실행

**사전 준비:** `painting-process-equipment-defect-detection-model-service`가 로컬 환경(`http://localhost:8001`)에서 먼저 실행 중이어야 합니다.

1.  **Python 가상 환경 설정**:
    ```bash
    python -m venv venv
    # Windows
    source venv/Scripts/activate
    # macOS/Linux
    # source venv/bin/activate
    ```

2.  **의존성 설치**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **.env 파일 설정**:
    프로젝트 루트에 `.env` 파일을 생성하고, Azure Storage 연결 문자열을 추가합니다.
    ```env
    AZURE_CONNECTION_STRING="<Your_Azure_Storage_Connection_String>"
    ```

4.  **애플리케이션 실행**:
    ```bash
    # 포트 8008에서 실행
    uvicorn app.main:app --host 0.0.0.0 --port 8008 --reload
    ```
    실행 후 `http://localhost:8008/docs`에서 API 문서를 확인할 수 있습니다.

### 4.2. Docker를 이용한 실행

**사전 준비:** `painting-process-equipment-defect-detection-model-service`가 `model-service`라는 컨테이너 이름으로 동일한 Docker 네트워크(`smart-fast-net`)에서 실행 중이어야 합니다.

1.  **Docker 네트워크 생성** (이미 생성했다면 생략):
    ```bash
    docker network create smart-fast-net
    ```

2.  **모델 서비스 실행** (이미 실행 중이라면 생략):
    ```bash
    # 모델 서비스 디렉토리에서 실행
    docker build -t model-service .
    docker run --name model-service --network smart-fast-net -p 8001:8001 model-service
    ```

3.  **시뮬레이터 서비스 Docker 이미지 빌드**:
    `.env` 파일이 현재 디렉토리에 있는지 확인한 후, 아래 명령어를 실행합니다. (`Dockerfile`이 `.env` 파일을 이미지 안으로 복사합니다.)
    ```bash
    docker build -t simulator-service .
    ```

4.  **시뮬레이터 서비스 Docker 컨테이너 실행**:
    `-p` 옵션으로 포트를 매핑하고, `-e` 옵션으로 모델 서비스의 주소를 컨테이너 이름으로 알려줍니다.
    ```bash
    docker run --name simulator-service --network smart-fast-net \
      -p 8008:8008 \
      -e PAINTING_SERVICE_URL="http://model-service:8001" \
      simulator-service
    ```

5.  **로그 확인**:
    ```bash
    # 시뮬레이터 로그 확인
    docker logs -f simulator-service
    ```

## 5. API 엔드포인트

서비스가 시작되면 `http://localhost:8008/docs` (또는 Docker IP)에서 API 문서를 통해 아래 엔드포인트를 테스트할 수 있습니다.

| HTTP Method | Endpoint                          | Description                                |
| :---------- | :-------------------------------- | :----------------------------------------- |
| `POST`      | `/simulator/start`                | 데이터 시뮬레이션을 시작합니다.            |
| `POST`      | `/simulator/stop`                 | 실행 중인 시뮬레이션을 중지합니다.         |
| `GET`       | `/simulator/status`               | 현재 스케줄러의 상태를 확인합니다.         |
| `POST`      | `/test/azure-storage-connection`  | Azure Blob Storage 연결을 테스트합니다.    |
| `POST`      | `/test/models-connection`         | 모델 서비스와의 연결을 테스트합니다.       |
```text
```