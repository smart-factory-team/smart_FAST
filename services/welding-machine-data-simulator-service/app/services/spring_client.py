import asyncio
import httpx
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from app.config.settings import settings

logger = logging.getLogger(__name__)


class SpringBootClient:
    """스프링부트 서비스와 통신하는 HTTP 클라이언트 (게이트웨이 경유)"""

    def __init__(self):
        # spring_boot_service_url → gateway_service_url 변경
        self.base_url = settings.gateway_service_url
        self.timeout = settings.spring_boot_timeout
        self.max_retries = settings.spring_boot_max_retries

    async def send_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        센서 데이터를 스프링부트 서비스로 전송 (게이트웨이 경유, MSAez 스타일)

        Args:
            sensor_data: 센서 데이터 (signal_type, values, machine_id, timestamp 포함)

        Returns:
            스프링부트 서비스 응답

        Raises:
            httpx.HTTPError: HTTP 요청 실패시
        """
        url = settings.spring_boot_endpoints["welding_data"]

        # ✅ SensorDataRequest DTO 형식으로 데이터 구성 (스프링부트가 기대하는 형식)
        payload = {
            # String 그대로
            "machineId": sensor_data.get("machine_id", "WELDING_MACHINE_001"),
            # timestamp (소문자)
            "timestamp": sensor_data.get("timestamp", datetime.now().isoformat()),
            "signalType": sensor_data["signal_type"],  # signalType 필드 추가
            "sensorValues": sensor_data["values"],     # sensorValues 배열 그대로 전송
            "dataSource": "simulator"
        }

        retry_count = 0
        last_exception = None

        while retry_count <= self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    logger.info(
                        f"게이트웨이를 통해 스프링부트로 센서 데이터 전송 시도 (시도 {retry_count + 1}/{self.max_retries + 1})")
                    logger.info(f"URL: {url}")
                    # DEBUG -> INFO로 변경하여 로그에서 확인
                    logger.info(f"Payload: {payload}")

                    headers = {
                        "Content-Type": "application/json",
                        "X-Source": "welding-simulator",
                        "X-Gateway-Route": "weldingprocessmonitoring"
                    }

                    response = await client.post(url, json=payload, headers=headers)

                    # HTTP 상태 코드 확인
                    response.raise_for_status()

                    result = response.json()
                    logger.info(f"게이트웨이를 통한 스프링부트 응답 성공: {result}")

                    return result

            except asyncio.CancelledError:
                logger.info("HTTP 요청이 취소되었습니다.")
                raise  # CancelledError는 다시 raise해야 함

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    f"게이트웨이 타임아웃 (시도 {retry_count + 1}): {str(e)}")

            except httpx.HTTPStatusError as e:
                last_exception = e
                logger.error(
                    f"게이트웨이 HTTP 오류 (시도 {retry_count + 1}): {e.response.status_code} - {e.response.text}")

                # 4xx 오류는 재시도하지 않음
                if 400 <= e.response.status_code < 500:
                    raise e

            except httpx.RequestError as e:
                last_exception = e
                logger.error(
                    f"게이트웨이 연결 오류 (시도 {retry_count + 1}): {str(e)}")

            except Exception as e:
                last_exception = e
                logger.error(
                    f"게이트웨이 예상치 못한 오류 (시도 {retry_count + 1}): {str(e)}")

            retry_count += 1

            # 마지막 시도가 아니면 잠시 대기
            if retry_count <= self.max_retries:
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("재시도 대기 중 작업 취소됨")
                    raise

        # 모든 재시도 실패
        logger.error(f"게이트웨이를 통한 스프링부트 전송 실패 - 모든 재시도 완료")
        raise last_exception

    def _extract_machine_id(self, machine_id_str: str) -> int:
        """기계 ID에서 숫자만 추출"""
        import re
        numbers = re.findall(r'\d+', machine_id_str)
        return int(numbers[0]) if numbers else 1

    def _format_timestamp(self, timestamp_str: str) -> str:
        """타임스탬프를 Java Date 형식으로 변환 (짧은 형식 사용)"""
        try:
            # 스프링부트 예시에서는 "2025-08-19" 같은 짧은 형식 사용
            from datetime import datetime
            if isinstance(timestamp_str, str):
                # 'Z' 또는 '+00:00' 제거하고 파싱
                clean_timestamp = timestamp_str.replace(
                    'Z', '').replace('+00:00', '')
                dt = datetime.fromisoformat(clean_timestamp)
            else:
                dt = datetime.now()

            # 짧은 날짜 형식으로 변환 (예시와 동일하게): "2025-08-19"
            return dt.strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"타임스탬프 변환 실패: {e}, 현재 날짜 사용")
            return datetime.now().strftime('%Y-%m-%d')

    def _map_sensor_values_to_db_columns(self, values: list) -> Dict[str, float]:
        """센서 값 배열을 DB 컬럼에 매핑 (기존 엔티티 구조에 맞게)"""
        def get_value_at_index(index: int) -> Optional[float]:
            return float(values[index]) if index < len(values) else None

        return {
            "sensorValue0Ms": get_value_at_index(0),
            "sensorValue25Ms": get_value_at_index(25),
            "sensorValue125Ms": get_value_at_index(125),
            "sensorValue312Ms": get_value_at_index(312),
            "sensorValue375Ms": get_value_at_index(375),
            "sensorValue625Ms": get_value_at_index(625),
            "sensorValue938Ms": get_value_at_index(938),
            "sensorValue1562Ms": get_value_at_index(1562),
            "sensorValue1875Ms": get_value_at_index(1875),
            "sensorValue2188Ms": get_value_at_index(2188),
            "sensorValue2812Ms": get_value_at_index(2812),
            "sensorValue3125Ms": get_value_at_index(3125),
            "sensorValue3438Ms": get_value_at_index(3438),
            "sensorValue4062Ms": get_value_at_index(4062),
        }

    async def health_check(self) -> bool:
        """
        스프링부트 서비스 헬스 체크 (게이트웨이 경유)

        Returns:
            서비스 상태 (True: 정상, False: 비정상)
        """
        try:
            url = settings.spring_boot_endpoints["health"]

            async with httpx.AsyncClient(timeout=5) as client:
                headers = {
                    "X-Source": "welding-simulator",
                    "X-Gateway-Route": "weldingprocessmonitoring"
                }

                response = await client.get(url, headers=headers)

                if response.status_code == 200:
                    try:
                        health_data = response.json()
                        status = health_data.get("status", "").upper()
                        logger.info(f"게이트웨이를 통한 스프링부트 헬스 체크 성공: {status}")
                        return status == "UP"
                    except:
                        # JSON 파싱 실패 시에도 200이면 정상으로 간주
                        logger.info("게이트웨이를 통한 스프링부트 헬스 체크 성공 (200 OK)")
                        return True
                else:
                    logger.warning(
                        f"게이트웨이를 통한 스프링부트 헬스 체크 실패: HTTP {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"게이트웨이를 통한 스프링부트 헬스 체크 오류: {str(e)}")
            return False


# 전역 클라이언트 인스턴스
spring_client = SpringBootClient()
