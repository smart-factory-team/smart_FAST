import sys
import os

# 현재 서비스의 루트 디렉토리를 Python 경로에 추가
service_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if service_root not in sys.path:
    sys.path.insert(0, service_root)

print(f"Added to Python path: {service_root}")  # 디버깅용