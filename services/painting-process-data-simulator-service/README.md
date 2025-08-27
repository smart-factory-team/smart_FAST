# Painting Process Backend Simulator Service

## 1. 서비스 소개
이 서비스는 Spring Boot 백엔드 애플리케이션으로 공정 데이터를 전송하는 시뮬레이터입니다.

주기적으로 Azure Blob Storage에 저장된 CSV 데이터를 읽어와, Spring Boot 백엔드의 API를 호출하여 데이터를 전송하는 역할을 합니다.

## 2. 주요 기능
- **주기적 데이터 전송**: `APScheduler`를 사용하여 설정된 시간 간격마다 Azure의 데이터를 백엔드로 전송합니다.
- **Azure Blob Storage 연동**: Azure Blob Storage에 저장된 CSV 파일을 읽어 시뮬레이션에 사용합니다.
- **Spring Boot 백엔드 연동**: `HTTPX` 클라이언트를 사용하여 Spring Boot 백엔드의 API를 비동기적으로 호출합니다.

## 3. 프로젝트 구조

```text
painting-process-data-simulator-service/
├── app/
│   ├── main.py                      # FastAPI 애플리케이션의 메인 진입점
│   ├── config/
│   │   ├── settings.py              # Pydantic-settings를 이용한 환경 변수 및 설정 관리
│   │   └── logging_config.py        # 서비스 로깅 설정
│   ├── routers/
│   │   └── simulator_router.py      # 시뮬레이터 시작/중지/상태 확인 API
│   └── services/
│       ├── scheduler_service.py     # APScheduler를 사용한 핵심 스케줄링 로직
│       ├── backend_client.py        # Spring Boot 백엔드 API 호출 클라이언트
│       └── azure_storage.py         # Azure Blob Storage 데이터 처리 서비스
├── .env                             # Azure 연결 문자열 등 민감한 환경 변수 파일
├── Dockerfile                       # Docker 이미지 빌드 설정
├── requirements.txt                 # Python 라이브러리 의존성 목록
└── README.md                        # 서비스 개요 및 사용법 문서
```

## 4. 설치 및 실행 방법

### 로컬 환경에서 실행

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
    프로젝트 루트에 `.env` 파일을 생성하고, Azure Storage 연결 문자열과 백엔드 서비스 URL을 추가합니다.
    ```env
    AZURE_CONNECTION_STRING="<Your_Azure_Storage_Connection_String>"
    BACKEND_SERVICE_URL="<Your_Backend_Service_URL>"
    ```

4.  **애플리케이션 실행**:
    ```bash
    uvicorn app.main:app --reload --port 8011 
    ```

## 5. API 엔드포인트

서비스가 시작되면 다음 URL로 API 문서(Swagger UI)에 접근할 수 있습니다: `http://localhost:8011/docs`

| HTTP Method | Endpoint           | Description                  |
| :---------- | :----------------- | :--------------------------- |
| `POST`      | `/simulator/start` | 데이터 전송 시뮬레이션을 시작합니다. |
| `POST`      | `/simulator/stop`  | 실행 중인 시뮬레이션을 중지합니다. |
| `GET`       | `/simulator/status`| 현재 스케줄러의 상태를 확인합니다. |

## 6. 로깅 (Logging)

이 서비스는 `app/config/logging_config.py` 파일을 통해 중앙에서 로깅 설정을 관리합니다.

- **로그 형식**: 모든 로그는 `시간 - 모듈 - 로그 레벨 - 메시지` 형식으로 기록됩니다.
- **로그 출력**:
  - **콘솔**: 실시간으로 로그가 콘솔에 출력됩니다.
  - **파일**: 로그는 `logs/service.log` 파일에도 저장되어, 서비스 실행 이력을 확인할 수 있습니다.
