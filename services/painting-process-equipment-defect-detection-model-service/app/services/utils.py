import json
import os
from pydantic import BaseModel
from datetime import datetime

class IssueLogInput(BaseModel):
    machineId: str
    timeStamp: datetime # datetime 객체로 자동 변환
    thick: float
    voltage: float
    current: float
    temper: float

def save_issue_log(log_data: dict, config: dict): # config 인자 유지
    """
    분석된 이슈 로그 데이터를 파일에 저장합니다.
    로그 파일 경로는 설정 파일에서 읽어옵니다.

    Args:
        log_data (dict): 저장할 로그 데이터 (분석 결과)
        config (dict): 애플리케이션 설정 딕셔너리
    """
    # 설정이 로드되었는지 확인 (이제 인자로 받은 config 사용)
    if config is None:
         print("오류: 설정이 인자로 전달되지 않아 로그를 저장할 수 없습니다.")
         return

    # config에서 로그 파일 경로 가져오기 (인자로 받은 config 사용)
    log_file_path = config.get("logs", {}).get("file_path")

    if not log_file_path:
        print("경고: 설정 파일에 로그 저장 경로(logs.file_path)가 지정되지 않았습니다. 로그를 파일에 저장하지 않습니다.")
        return

    # 파일 경로가 존재하지 않으면 디렉토리 생성
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # JSON Lines 형식으로 파일에 로그 추가 ('a' 모드는 append)
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False)
            f.write('\n') # JSON Lines 형식으로 각 로그 뒤에 줄바꿈 추가
    except Exception as e:
        print(f"로그 파일 저장 중 오류 발생: {e}")