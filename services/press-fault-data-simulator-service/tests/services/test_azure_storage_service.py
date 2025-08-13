import asyncio
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, List, Optional
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import contextlib

# Note: Testing library and framework
# These tests are written for pytest with pytest-asyncio for asyncio test support.

# Attempt to import the service under test. We try common module paths.
# Adjust the import path if your actual module is different.
try:
    # If the service lives in app/services/azure_storage_service.py
    from app.services.azure_storage_service import AzureStorageService
except ImportError:
    # Fallback: The code snippet might be colocated; attempt relative import variants.
    try:
        from services.azure_storage_service import AzureStorageService  # type: ignore
    except ImportError:
        # As a last resort, define a sentinel to fail with a clear message.
        AzureStorageService = None  # type: ignore


pytestmark = pytest.mark.asyncio


def _require_service_class():
    if AzureStorageService is None:
        pytest.fail(
            "Could not import AzureStorageService. "
            "Update the import in test_azure_storage_service.py to the correct module path."
        )


@dataclass
class FakeBlobItem:
    name: str
    last_modified: datetime


class _AsyncIter:
    def __init__(self, items: List[Any]):
        self._items = items

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


def _make_service_with_mocks(
    *,
    list_blobs: Optional[List[FakeBlobItem]] = None,
    download_blob_bytes: Optional[bytes] = None,
    get_container_properties_raises: Optional[BaseException] = None,
) -> AzureStorageService:
    """
    Helper to instantiate AzureStorageService with patched BlobServiceClient and settings.
    """
    _require_service_class()

    # Patches
    blob_service_client_mock = MagicMock(name="BlobServiceClient")
    container_client_mock = MagicMock(name="ContainerClient")
    blob_client_mock = MagicMock(name="BlobClient")

    # connect() path
    if get_container_properties_raises:
        container_client_mock.get_container_properties = AsyncMock(
            side_effect=get_container_properties_raises
        )
    else:
        container_client_mock.get_container_properties = AsyncMock(return_value=None)

    # get_csv_files_list path
    if list_blobs is not None:
        container_client_mock.list_blobs.return_value = _AsyncIter(list_blobs)
    else:
        container_client_mock.list_blobs.return_value = _AsyncIter([])

    # load_csv_chunk_from_storage path
    download_stream_mock = MagicMock()
    download_stream_mock.readall = AsyncMock(
        return_value=download_blob_bytes if download_blob_bytes is not None else b""
    )
    blob_client_mock.download_blob = AsyncMock(return_value=download_stream_mock)

    blob_service_client_mock.get_container_client.return_value = container_client_mock
    blob_service_client_mock.get_blob_client.return_value = blob_client_mock
    blob_service_client_mock.close = AsyncMock(return_value=None)

    # settings and logger patches
    settings_stub = SimpleNamespace(
        AZURE_STORAGE_CONNECTION_STRING="UseDevelopmentStorage=true",
        AZURE_STORAGE_CONTAINER_NAME="test-container",
        PRESS_FAULT_FOLDER="press/fault",
    )

    logger_mock = MagicMock()
    logger_mock.info = MagicMock()
    logger_mock.warning = MagicMock()
    logger_mock.error = MagicMock()

    # Patch constructors and settings in the service module namespace
    # We patch where they are looked up: inside the module under test.
    patches = [
        patch(
            "app.services.azure_storage_service.BlobServiceClient.from_connection_string",
            return_value=blob_service_client_mock,
        ),
        patch("app.services.azure_storage_service.settings", settings_stub),
        patch("app.services.azure_storage_service.system_log", logger_mock),
        patch("app.services.azure_storage_service.pd.read_csv"),
    ]

    # If import path differs, try alternative patch targets too (best effort).
    alt_patches = [
        patch(
            "services.azure_storage_service.BlobServiceClient.from_connection_string",
            return_value=blob_service_client_mock,
        ),
        patch("services.azure_storage_service.settings", settings_stub),
        patch("services.azure_storage_service.system_log", logger_mock),
        patch("services.azure_storage_service.pd.read_csv"),
    ]

    # Activate patches (first set, then fallback if they fail)
    active_patches = []
    for p in patches:
        try:
            active_patches.append(p.start())
        except Exception:
            # Try alternative location
            alt = alt_patches[patches.index(p)]
            try:
                active_patches.append(alt.start())
            except Exception:
                # If both fail, re-raise to surface import path issue
                raise

    # Build service instance
    service = AzureStorageService()

    # Attach mocks to service for assertions
    service.__test_mocks__ = SimpleNamespace(
        blob_service_client=blob_service_client_mock,
        container_client=container_client_mock,
        blob_client=blob_client_mock,
        logger=logger_mock,
        read_csv=active_patches[-1],  # The last patch corresponds to pd.read_csv
        patches=patches,
        active_patches=active_patches,
    )
    return service


def _cleanup_service_patches(service: Any):
    for p in getattr(service.__test_mocks__, "active_patches", []):
        with contextlib.suppress(Exception):
            p.stop()


@pytest.fixture
def df_required_cols():
    return pd.DataFrame(
        {
            "AI0_Vibration": [1, 2, 3],
            "AI1_Vibration": [4, 5, 6],
            "AI2_Current": [7, 8, 9],
        }
    )


async def test_connect_success():
    service = _make_service_with_mocks()
    try:
        ok = await service.connect()
        assert ok is True
        assert service.is_connected is True
        # verify calls
        service.__test_mocks__.blob_service_client.get_container_client.assert_called_once_with(
            "test-container"
        )
        service.__test_mocks__.container_client.get_container_properties.assert_awaited()
        service.__test_mocks__.logger.info.assert_any_call("Azure Storage 연결 성공")
    finally:
        await service.close()
        _cleanup_service_patches(service)


async def test_connect_failure_logs_and_returns_false():
    service = _make_service_with_mocks(
        get_container_properties_raises=RuntimeError("boom")
    )
    try:
        ok = await service.connect()
        assert ok is False
        assert service.is_connected is False
        # error log emitted
        assert any(
            "Azure Storage 연결 실패" in str(call.args[0])
            for call in service.__test_mocks__.logger.error.call_args_list
        )
    finally:
        await service.close()
        _cleanup_service_patches(service)


async def test_get_csv_files_list_filters_and_sorts_by_last_modified():
    now = datetime.now(timezone.utc)
    files = [
        FakeBlobItem(
            name="press/fault/a.csv", last_modified=now - timedelta(minutes=2)
        ),
        FakeBlobItem(
            name="press/fault/b.csv", last_modified=now - timedelta(minutes=1)
        ),
        FakeBlobItem(
            name="press/fault/sub/c.csv", last_modified=now
        ),  # subfolder excluded
        FakeBlobItem(name="press/fault/d.txt", last_modified=now),  # non-csv excluded
    ]
    service = _make_service_with_mocks(list_blobs=files)
    try:
        names = await service.get_csv_files_list()
        # Only a.csv and b.csv, sorted latest first => b.csv then a.csv
        assert names == ["press/fault/b.csv", "press/fault/a.csv"]
        service.__test_mocks__.logger.info.assert_any_call(
            f"CSV 파일 목록 조회 완료: {len(names)}개 파일"
        )
    finally:
        await service.close()
        _cleanup_service_patches(service)


async def test_get_csv_files_list_exception_returns_empty_and_logs():
    # Make list_blobs raise when iterating by replacing container_client.list_blobs with one that raises
    service = _make_service_with_mocks()

    # Replace list_blobs with an iterator that raises
    async def _raiser(**kwargs):
        raise RuntimeError("list error")

    service.__test_mocks__.container_client.list_blobs.side_effect = _raiser
    try:
        names = await service.get_csv_files_list()
        assert names == []
        assert any(
            "CSV 파일 목록 조회 실패" in str(call.args[0])
            for call in service.__test_mocks__.logger.error.call_args_list
        )
    finally:
        await service.close()
        _cleanup_service_patches(service)


async def test_load_csv_chunk_success_calls_pandas_with_skiprows_and_nrows(
    df_required_cols,
):
    csv_bytes = b"AI0_Vibration,AI1_Vibration,AI2_Current\n1,4,7\n2,5,8\n3,6,9\n"
    service = _make_service_with_mocks(download_blob_bytes=csv_bytes)
    try:
        # Arrange pd.read_csv to return a DataFrame with required columns
        with patch.object(
            pd, "read_csv", return_value=df_required_cols
        ) as read_csv_mock:
            df = await service.load_csv_chunk_from_storage(
                file_name="press/fault/a.csv", start_row=1200, num_rows=600
            )
        assert df is df_required_cols
        # Verify pd.read_csv called with expected skiprows and nrows
        args, kwargs = read_csv_mock.call_args
        assert "skiprows" in kwargs and "nrows" in kwargs
        # skiprows=range(1, start_row+1), nrows=num_rows
        # Verify the skiprows range
        skiprows_range = kwargs["skiprows"]
        assert skiprows_range.start == 1
        assert skiprows_range.stop == 1201  # This will skip rows 1–1200
        assert kwargs["nrows"] == 600
    finally:
        await service.close()
        _cleanup_service_patches(service)


async def test_load_csv_chunk_missing_required_columns_returns_none():
    # DataFrame with missing required columns
    df_missing = pd.DataFrame({"AI0_Vibration": [1], "AI1_Vibration": [2]})
    csv_bytes = b"AI0_Vibration,AI1_Vibration\n1,2\n"
    service = _make_service_with_mocks(download_blob_bytes=csv_bytes)
    try:
        with patch.object(pd, "read_csv", return_value=df_missing):
            df = await service.load_csv_chunk_from_storage(
                file_name="press/fault/a.csv", start_row=0, num_rows=10
            )
        assert df is None
        assert any(
            "필수 컬럼 누락" in str(call.args[0])
            for call in service.__test_mocks__.logger.error.call_args_list
        )
    finally:
        await service.close()
        _cleanup_service_patches(service)


async def test_load_csv_chunk_exception_returns_none_and_logs():
    service = _make_service_with_mocks(download_blob_bytes=b"bad,csv\n")
    try:
        with patch.object(pd, "read_csv", side_effect=ValueError("parse error")):
            df = await service.load_csv_chunk_from_storage(
                file_name="press/fault/a.csv", start_row=0, num_rows=10
            )
        assert df is None
        assert any(
            "CSV 청크 로드 실패" in str(call.args[0])
            for call in service.__test_mocks__.logger.error.call_args_list
        )
    finally:
        await service.close()
        _cleanup_service_patches(service)


async def test_get_next_minute_data_happy_path_returns_chunk_and_updates_index(
    df_required_cols,
):
    # Provide a chunk shorter than row_per_minute? For happy path (not end), make equal length
    chunk_len = 600
    df = pd.DataFrame(
        {
            "AI0_Vibration": list(range(chunk_len)),
            "AI1_Vibration": list(range(chunk_len)),
            "AI2_Current": list(range(chunk_len)),
        }
    )
    files = [
        FakeBlobItem(
            name="press/fault/latest.csv", last_modified=datetime.now(timezone.utc)
        )
    ]
    service = _make_service_with_mocks(list_blobs=files, download_blob_bytes=b"dummy")
    try:
        with patch.object(pd, "read_csv", return_value=df):
            result = await service.get_next_minute_data()
        assert result is not None
        chunk, filename, is_eof = result
        assert filename == "press/fault/latest.csv"
        assert isinstance(chunk, pd.DataFrame)
        assert len(chunk) == chunk_len
        assert is_eof is False
        # current index advanced
        assert service.current_row_index == chunk_len
        # status reflects processing state
        status = service.get_current_status()
        assert status["status"] == "processing"
        assert status["current_file"] == "press/fault/latest.csv"
        assert status["current_index"] == chunk_len
        assert status["rows_per_chunk"] == service.row_per_minute
    finally:
        await service.close()
        _cleanup_service_patches(service)


@pytest.mark.xfail(
    raises=AttributeError,
    reason="get_next_minute_data calls undefined self.get_next_chunk() when an empty chunk is returned.",
)
async def test_get_next_minute_data_empty_chunk_path_raises_due_to_bug():
    # Force an empty DataFrame to trigger the len(chunk_data) == 0 branch.
    empty_df = pd.DataFrame(
        {"AI0_Vibration": [], "AI1_Vibration": [], "AI2_Current": []}
    )
    files = [
        FakeBlobItem(
            name="press/fault/latest.csv", last_modified=datetime.now(timezone.utc)
        )
    ]
    service = _make_service_with_mocks(list_blobs=files, download_blob_bytes=b"")
    try:
        with patch.object(pd, "read_csv", return_value=empty_df):
            # This is expected to raise because the code calls self.get_next_chunk(), which doesn't exist.
            await service.get_next_minute_data()
    finally:
        await service.close()
        _cleanup_service_patches(service)


async def test_get_next_minute_data_none_chunk_returns_none(df_required_cols):
    # If load_csv_chunk_from_storage returns None, the method should return None and log an error
    files = [
        FakeBlobItem(
            name="press/fault/latest.csv", last_modified=datetime.now(timezone.utc)
        )
    ]
    service = _make_service_with_mocks(
        list_blobs=files, download_blob_bytes=b"irrelevant"
    )
    try:
        with patch.object(pd, "read_csv", side_effect=ValueError("parse error")):
            # load_csv_chunk_from_storage will catch and return None
            result = await service.get_next_minute_data()
        assert result is None
        assert any(
            "데이터 로드 실패" in str(call.args[0])
            for call in service.__test_mocks__.logger.error.call_args_list
        )
    finally:
        await service.close()
        _cleanup_service_patches(service)


async def test_move_to_next_file_resets_state():
    service = _make_service_with_mocks()
    try:
        # Simulate some state
        service.current_file_name = "press/fault/old.csv"
        service.current_row_index = 123
        service._move_to_next_file()
        assert service.current_file_name is None
        assert service.current_row_index == 0
    finally:
        await service.close()
        _cleanup_service_patches(service)


def test_get_current_status_no_file_loaded():
    service = _make_service_with_mocks()
    try:
        service._move_to_next_file()  # ensure reset
        status = service.get_current_status()
        assert status == {"status": "no_file_loaded"}
    finally:
        asyncio.get_event_loop().run_until_complete(service.close())
        _cleanup_service_patches(service)


def test_clear_cache_resets_state():
    service = _make_service_with_mocks()
    try:
        service.current_file_name = "press/fault/something.csv"
        service.current_row_index = 999
        service._clear_cache()
        assert service.current_file_name is None
        assert service.current_row_index == 0
    finally:
        asyncio.get_event_loop().run_until_complete(service.close())
        _cleanup_service_patches(service)


async def test_close_closes_underlying_client():
    service = _make_service_with_mocks()
    try:
        await service.close()
        service.__test_mocks__.blob_service_client.close.assert_awaited()
    finally:
        _cleanup_service_patches(service)
