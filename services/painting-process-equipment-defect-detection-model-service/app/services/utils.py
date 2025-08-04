import json
import os
from datetime import datetime
from pydantic import BaseModel, Field

class IssueLogInput(BaseModel):
    machineId: str
    timeStamp: datetime
    thick: float
    voltage: float
    current: float
    temper: float
    issue: str = Field(default="PAINT-EQ-UNKNOWN")
    isSolved: bool = Field(default=False)

def save_issue_log(log_data: dict, config: dict):
    """
    이슈 로그 데이터를 JSON Lines 형식으로 파일에 저장합니다.
    """
    if not config or not config.get("logs", {}).get("file_path"):
        # 수정된 print 문
        print("경고: 설정 파일에 로그 저장 경로(logs.file_path)가 지정되지 않았습니다.")
        return

    log_file_path = config["logs"]["file_path"]
    log_dir = os.path.dirname(log_file_path)

    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError:
            # 수정된 print 문
            print(f"로그 파일 디렉토리 생성 중 오류 발생: {log_dir}")
            return

    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False)
            f.write("\n")
    except OSError:
        # 수정된 print 문
        print(f"로그 파일 저장 중 오류 발생: {log_file_path}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")