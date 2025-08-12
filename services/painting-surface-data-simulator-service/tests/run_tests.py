#!/usr/bin/env python3
"""
테스트 실행 스크립트
사용법: python run_tests.py [옵션]
"""

import subprocess
import sys
import os


def run_command(command, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ 성공!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 실패: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """메인 함수"""
    print("🧪 Painting Surface Data Simulator Service 테스트 실행")
    print("=" * 60)
    
    # 현재 디렉토리 확인
    if not os.path.exists("tests"):
        print("❌ tests 폴더를 찾을 수 없습니다.")
        print("   이 스크립트는 프로젝트 루트에서 실행해야 합니다.")
        sys.exit(1)
    
    # 테스트 의존성 설치 확인
    print("📦 테스트 의존성 확인 중...")
    try:
        import pytest
        import httpx
        print("✅ 필요한 패키지가 설치되어 있습니다.")
    except ImportError as e:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("   다음 명령어로 설치하세요:")
        print("   pip install -r requirements-dev.txt")
        sys.exit(1)
    
    # 테스트 실행
    success = True
    
    # 1. 기본 테스트 실행
    success &= run_command(
        "py -3.10 -m pytest tests/ -v",
        "기본 테스트 실행"
    )
    
    # 2. 커버리지 테스트 실행
    success &= run_command(
        "py -3.10 -m pytest tests/ --cov=app --cov-report=term-missing",
        "코드 커버리지 테스트 실행"
    )
    
    # 3. 특정 테스트 파일 실행 (예시)
    if success:
        print("\n📋 개별 테스트 파일 실행 예시:")
        print("   py -3.10 -m pytest tests/test_settings.py -v")
        print("   py -3.10 -m pytest tests/test_logger.py -v")
        print("   py -3.10 -m pytest tests/test_azure_storage.py -v")
        print("   py -3.10 -m pytest tests/test_model_client.py -v")
        print("   py -3.10 -m pytest tests/test_scheduler_service.py -v")
        print("   py -3.10 -m pytest tests/test_simulator_router.py -v")
        print("   py -3.10 -m pytest tests/test_main.py -v")
    
    # 결과 요약
    print(f"\n{'='*60}")
    if success:
        print("🎉 모든 테스트가 성공적으로 실행되었습니다!")
    else:
        print("⚠️ 일부 테스트 실행에 실패했습니다.")
        print("   위의 오류 메시지를 확인하고 수정하세요.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
