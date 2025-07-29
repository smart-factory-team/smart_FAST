# SMART FAST - AI 기반 제조업 결함 탐지 플랫폼

자동차 제조 공정 전반에서 발생할 수 있는 **설비 고장 및 품질 결함 탐지**를 위한 MSA 기반 AI 서비스 플랫폼입니다.

## 🎯 목적

* 공정별 AI 모델을 **독립적인 마이크로서비스**로 구성하여 확장성과 유지보수성 확보
* **Spring Cloud Gateway**와 **FastAPI**를 연동한 하이브리드 MSA 아키텍처 구현
* 실제 스마트팩토리 환경을 가정하여 **현장 적용 가능성**을 고려한 서비스 설계
* **에이전트 기반 모델 조합**을 통한 복합 AI 작업 지원

## 🏗️ 아키텍처

- **Spring Cloud Gateway**: API 라우팅, 인증, 로드밸런싱
- **FastAPI 마이크로서비스**: 각 AI 모델별 독립 서비스
- **Docker**: 컨테이너 기반 배포 및 오케스트레이션
- **Agent Service**: 여러 모델 조합 및 복합 AI 작업

## 📁 프로젝트 구조

```
SMART_FAST/
├── services/
│   ├── painting-process-equipment-defect-detection-model-service/   # 도장 공정 장비 결함 탐지
│   ├── painting-surface-defect-detection-model-service/            # 도장 표면 결함 탐지  
│   ├── press-defect-detection-model-service/                       # 프레스 결함 탐지
│   ├── press-fault-detection-model-service/                        # 프레스 고장 탐지
│   ├── vehicle-assembly-process-defect-detection-model-service/     # 차량 조립 공정 결함 탐지
│   ├── welding-machine-defect-detection-model-service/             # 용접기 결함 탐지
│   ├── agent-service/                                              # 에이전트 서비스
│   └── shared/                                                     # 공통 라이브러리
├── infrastructure/                                                 # Docker 및 배포 설정
├── .vscode/                                                       # VSCode 개발 환경 설정
└── venv/                                                          # Python 가상환경
```

각 서비스 폴더는 다음 구조로 구성:
```
service-name/
├── app/
│   ├── main.py          # FastAPI 애플리케이션
│   ├── models/          # AI 모델 정의
│   ├── routers/         # API 라우터
│   └── services/        # 비즈니스 로직
├── Dockerfile           # Docker 이미지 설정
├── requirements.txt     # Python 의존성
└── README.md           # 서비스별 문서
```

## 👥 팀원별 담당 서비스

| 팀원 | 담당 서비스 | 포트 | 설명 |
|------|-------------|------|------|
| 김태현 | painting-process-equipment-defect-detection-model-service | 8001 | 도장 공정 장비 결함 탐지 |
| 이원욱 | painting-surface-defect-detection-model-service | 8002 | 도장 표면 결함 탐지 |
| 김해연 | press-defect-detection-model-service | 8003 | 프레스 결함 탐지 |
| 배소연 | press-fault-detection-model-service | 8004 | 프레스 고장 탐지 |
| 권도윤 | vehicle-assembly-process-defect-detection-model-service | 8005 | 차량 조립 공정 결함 탐지 |
| 한다현 | welding-machine-defect-detection-model-service | 8006 | 용접기 결함 탐지 |
| 김승환 | agent-service | 8007 | 에이전트 서비스 (모델 조합) |

## ⚙️ 기술 스택

* **Backend**: FastAPI, Python 3.9+
* **Gateway**: Spring Cloud Gateway
* **Container**: Docker, Docker Compose
* **AI/ML**: TensorFlow, PyTorch, OpenCV, Scikit-learn
* **Monitoring**: Prometheus, Grafana (선택사항)
* **Development**: VSCode, Black (코드 포맷터)

## 🚀 개발 시작하기

### 1. 환경 설정
```bash
# 레포지토리 클론
git clone <repository-url>
cd SMART_FAST

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# source venv/Scripts/activate  # bash

# 기본 의존성 설치
pip install -r requirements-dev.txt
```

### 2. 개별 서비스 개발
각 팀원은 자신의 담당 서비스 디렉토리에서 작업:

```bash
# 예: 도장 공정 장비 결함 탐지 모델 담당자
cd services/painting-process-equipment-defect-detection-model-service

# 필요한 AI/ML 라이브러리 추가
pip install torch torchvision opencv-python numpy pandas scikit-learn

# 의존성 파일 업데이트
pip freeze > requirements.txt

# 모델 구현
# app/main.py 파일의 predict() 함수에 실제 AI 모델 로직 구현
```

### 3. 개별 서비스 실행 및 테스트
```bash
# 개별 서비스 실행
cd services/your-service-name
uvicorn app.main:app --reload --port 800X

# 또는 Docker로 실행
docker build -t your-service-name .
docker run -p 800X:800X your-service-name

# API 문서 확인
# http://localhost:800X/docs
```

### 4. 전체 서비스 실행
```bash
# 모든 서비스 Docker Compose로 실행
cd infrastructure
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build
```

## 📋 개발 가이드라인

### API 엔드포인트 구조
각 서비스는 다음과 같은 기본 엔드포인트를 제공해야 합니다:

- `GET /` - 서비스 정보
- `GET /health` - 헬스 체크
- `POST /predict` - AI 모델 예측
- `POST /predict/file` - 파일 업로드를 통한 예측
- `GET /model/info` - 모델 정보 조회

### 개발 시 주의사항
1. **포트 충돌 방지**: 각자 할당된 포트 사용
2. **공통 라이브러리 활용**: `services/shared/` 디렉토리의 공통 코드 사용
3. **Docker 테스트**: 개발 완료 후 반드시 Docker로 실행 테스트
4. **API 문서화**: FastAPI 자동 문서화 활용 (`/docs` 엔드포인트)
5. **의존성 관리**: 새로운 라이브러리 설치 후 `pip freeze > requirements.txt`

## 🔧 Git 브랜치 전략 및 커밋 컨벤션

본 프로젝트는 **Trunk-Based Development (TBD)** 전략을 따릅니다. 브랜치는 단기 브랜치만 운영하며, 작업 후 빠르게 `main`에 병합합니다.

### 📌 브랜치 전략
* 기본 브랜치: `main`
* 작업 브랜치: 단기 브랜치 → PR → merge 후 삭제
* 각 브랜치는 명확한 작업 단위를 기준으로 생성

### 📄 브랜치 네이밍 규칙
형식: `<type>/<issue-number>/<short-description>`
* `type`: 작업 유형 (init, feat, fix, refactor, chore 등)
* `issue-number`: 연동된 이슈 번호 (선택사항)
* `short-description`: 작업 요약 (kebab-case)

### 💡 브랜치 예시
```
init/1/setup-msa-environment
feat/1/painting-process-model
feat/1/press-defect-detection
fix/1/docker-port-conflict
refactor/2/agent-service-architecture
chore/3/update-dependencies
```

### ✏️ 커밋 메시지 컨벤션
커밋 메시지는 **Gitmoji**를 사용하여 의미를 명확히 합니다.
형식: `:gitmoji: <type>: <commit message>`

### 🗂 주요 Gitmoji 목록
| 이모지 | 코드 | 의미 |
|--------|------|------|
| 🎉 | `:tada:` | 초기 커밋 |
| ✨ | `:sparkles:` | 새 기능 추가 |
| 🐛 | `:bug:` | 버그 수정 |
| ♻️ | `:recycle:` | 리팩토링 |
| 🔥 | `:fire:` | 코드/파일 제거 |
| 📝 | `:memo:` | 문서 수정 |
| 🔧 | `:wrench:` | 설정 파일 수정 |
| 🚑 | `:ambulance:` | 긴급 수정 |
| ✅ | `:white_check_mark:` | 테스트 추가 |
| 🚀 | `:rocket:` | 배포 관련 작업 |
| 📦 | `:package:` | 패키지 추가/변경 |
| 🐳 | `:whale:` | Docker 관련 작업 |
| 🔀 | `:twisted_rightwards_arrows:` | 브랜치 병합 |

### 💡 커밋 예시
```bash
git commit -m "🎉 init: MSA 기반 FastAPI 환경 초기 설정"
git commit -m "✨ feat: 도장 표면 결함 탐지 모델 구현"
git commit -m "🐛 fix: Docker 포트 충돌 문제 해결"
git commit -m "♻️ refactor: 에이전트 서비스 아키텍처 개선"
git commit -m "📝 docs: API 사용법 문서 추가"
git commit -m "🐳 docker: 프로덕션 환경 Docker 설정 추가"
```

### 🔄 PR 가이드
* 각 작업 완료 시 PR 생성
* PR 제목은 작업 내용을 명확히 표현
* 코드 리뷰 이후 `main`으로 merge
* 긴 브랜치 유지 지양, main은 항상 배포 가능한 상태 유지

## 🔧 주요 명령어

```bash
# 개발 환경 시작
make dev-start

# 서비스 테스트
make test

# 코드 포맷팅
black services/

# 전체 서비스 중지
docker-compose down

# 로그 확인
docker-compose logs -f service-name

# 특정 서비스 재시작
docker-compose restart service-name
```

## 📞 문의 및 협업

- 각 서비스는 독립적으로 개발 가능
- 서비스 간 통신이 필요한 경우 agent-service 담당자와 협의
- Docker/인프라 관련 문제 시 인프라 담당자에게 문의
- API 스펙 변경 시 팀 전체에 공유

## ℹ️ 참고

* 브랜치 및 커밋 단위를 작게 유지해 충돌 방지
* GitHub Issues와 연동하여 작업 관리
* VSCode 설정 파일이 포함되어 있어 일관된 개발 환경 제공
* 의존성 충돌 방지를 위해 가상환경 사용 필수

