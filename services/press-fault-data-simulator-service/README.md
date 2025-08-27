#####  CI/CD 테스트 1 
# Press Fault Data Simulator Service 

Azure Storage의 CSV 데이터를 이용해 압력기 고장 예측 시뮬레이션을 수행하는 FastAPI 서비스입니다.

## 🔍 개요

이 서비스는 Azure Blob Storage에 저장된 산업용 압력기 데이터(진동, 전류)를 1분 단위로 읽어와서 고장 예측 API에 전송하고, 결과를 로깅하는 시뮬레이터입니다.

## ✨ 주요 기능

- **Azure Storage 연동**: 파일을 실시간으로 읽어와 처리
- **백그라운드 시뮬레이션**: 설정 가능한 간격으로 자동 실행
- **고장 예측 API 연동**: ML 모델과 연계하여 고장 감지
- **구조화된 로깅**: JSON 형식으로 예측 결과 저장
- **REST API**: 시뮬레이션 제어 및 상태 조회

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Azure Storage  │────│  Simulator API   │────│ Prediction API  │
│   (CSV Data)    │    │   (FastAPI)      │    │   (ML Model)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Log Files      │
                       │   (JSON Format)  │
                       └──────────────────┘
```

## 📁 프로젝트 구조

```
app/
├── config/
│   └── settings.py          # 환경설정 관리
├── models/
│   └── data_models.py       # 데이터 모델 정의
├── routers/
│   ├── simulator_router.py  # 시뮬레이션 API 엔드포인트
│   └── test_connection_router.py # 연결 테스트 API
├── services/
│   ├── azure_storage_service.py    # Azure Storage 연동
│   ├── prediction_api_service.py   # 외부 API 호출
│   └── scheduler_service.py        # 백그라운드 스케줄러
├── utils/
│   └── logger.py            # 로깅 시스템
└── main.py                  # FastAPI 애플리케이션 진입점
```

## 🚀 시작하기

### 1. 환경 설정

`.env` 파일을 생성하고 필요한 환경변수를 설정:

```env
# Azure Storage 설정
AZURE_STORAGE_CONNECTION_STRING=my_connection_string
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 서비스 실행

```bash
uvicorn app.main:app --reload --port 8014
```

## 📡 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 서비스 상태 확인 |
| POST | `/simulator/start` | 시뮬레이션 시작 |
| POST | `/simulator/stop` | 시뮬레이션 종료 |
| GET | `/simulator/status` | 시뮬레이션 상태 조회 |

## 📊 데이터 형식

### 입력 데이터 (CSV)
```csv
AI0_Vibration,AI1_Vibration,AI2_Current
1.234,2.345,3.456
...
```

### 예측 결과 로그 (JSON)
```json
{
  "timestamp": "2025-01-01T12:00:00",
  "service_name": "press fault detection",
  "message": "🚨 FAULT DETECTED - Prediction: fault, Probability: 0.8542"
}
```

## 📋 요구사항

- Python 3.10
- Azure Storage Account
- 예측 API 서버

## 📝 로그 파일

- **시스템 로그**: 콘솔 출력
- **예측 결과**: `./logs/press_fault_detections.json`
- **오류 로그**: `./logs/press_fault_errors.json`