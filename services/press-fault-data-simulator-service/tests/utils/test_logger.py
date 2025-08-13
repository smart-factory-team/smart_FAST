# Tests for logger utilities: LevelBasedFormatter, CustomJsonFormatter, setup_loggers
# Testing framework: pytest
# These tests validate formatting logic, handler configuration, and JSON fields, focusing on public behaviors.
# Note: We add the service root to sys.path to import the 'app' package of this service.

import io
import json
import logging
import os
import sys
import importlib
import contextlib
from pathlib import Path

import pytest

# Ensure the service root is on sys.path so `import app...` resolves to this service's app/ package.
SERVICE_ROOT = Path(__file__).resolve().parents[2]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

MODULE_UNDER_TEST = None

def _clear_logger(name: str):
    logger = logging.getLogger(name)
    logger.propagate = False
    for h in list(logger.handlers):
        try:
            logger.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()
        except Exception:
            pass
    logger.handlers = []
    return logger

@pytest.fixture(autouse=True)
def isolate_logging():
    # Ensure clean state for "system" and "prediction" loggers
    _clear_logger("system")
    _clear_logger("prediction")
    yield
    _clear_logger("system")
    _clear_logger("prediction")

@pytest.fixture
def fake_settings(tmp_path, monkeypatch):
    """
    Provide a fake settings object compatible with the module under test:
      - LOG_DIR: directory for logs
      - LOG_LEVEL: string like "INFO"
      - ERROR_LOG_FILE_PATH: path to JSON log file for prediction logger
    Also set required environment variables so app.config.settings can import without ValidationError.
    """
    # Ensure required env var for Settings() exists
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")

    log_dir = tmp_path / "logs"
    error_log = log_dir / "errors.log"

    class _S:
        LOG_DIR = str(log_dir)
        LOG_LEVEL = "INFO"
        ERROR_LOG_FILE_PATH = str(error_log)

    # Patch the settings object used by the module under test.
    # The module code imports: from app.config.settings import settings
    settings_mod = importlib.import_module("app.config.settings")
    monkeypatch.setattr(settings_mod, "settings", _S(), raising=False)

    return _S()

def import_logger_module(monkeypatch, fake_settings):
    """
    Import the logger module after ensuring settings are patched.
    """
    global MODULE_UNDER_TEST

    # Ensure a clean import to pick up patched settings
    if "app.utils.logger" in sys.modules:
        del sys.modules["app.utils.logger"]

    MODULE_UNDER_TEST = importlib.import_module("app.utils.logger")
    return MODULE_UNDER_TEST

def test_level_based_formatter_formats_simple_for_info(monkeypatch, fake_settings):
    mod = import_logger_module(monkeypatch, fake_settings)
    fmt = mod.LevelBasedFormatter()

    rec = logging.LogRecord(
        name="prediction", level=logging.INFO, pathname=__file__, lineno=123,
        msg="Normal operation", args=(), exc_info=None
    )
    out = fmt.format(rec)
    assert out == "✅ [INFO] Normal operation"

def test_level_based_formatter_formats_detailed_for_warning_and_above(monkeypatch, fake_settings):
    mod = import_logger_module(monkeypatch, fake_settings)
    fmt = mod.LevelBasedFormatter()

    warn_rec = logging.LogRecord(
        name="prediction", level=logging.WARNING, pathname=__file__, lineno=10,
        msg="Anomaly detected", args=(), exc_info=None
    )
    err_rec = logging.LogRecord(
        name="prediction", level=logging.ERROR, pathname=__file__, lineno=11,
        msg="Critical failure", args=(), exc_info=None
    )
    out_warn = fmt.format(warn_rec)
    out_err = fmt.format(err_rec)

    assert out_warn == "🚨 [WARNING]:\n   └─ Anomaly detected"
    assert out_err == "🚨 [ERROR]:\n   └─ Critical failure"

def _build_stream_logger(name: str, formatter: logging.Formatter):
    logger = _clear_logger(name)
    logger.setLevel(logging.DEBUG)
    stream = io.StringIO()
    sh = logging.StreamHandler(stream)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger, stream

def test_custom_json_formatter_adds_timestamp_and_service_name(monkeypatch, fake_settings):
    mod = import_logger_module(monkeypatch, fake_settings)
    formatter = mod.CustomJsonFormatter("%(timestamp)s %(service_name)s %(message)s")

    logger, stream = _build_stream_logger("prediction", formatter)
    logger.warning("json event")
    output = stream.getvalue().strip()

    # Parse JSON record
    data = json.loads(output)
    assert data["service_name"] == "press fault detection"
    # Timestamp should be ISO-8601-ish string: presence of 'T' is a simple indicator
    assert isinstance(data["timestamp"], str) and "T" in data["timestamp"]
    assert data["message"] == "json event"

def test_custom_json_formatter_preserves_existing_timestamp(monkeypatch, fake_settings):
    mod = import_logger_module(monkeypatch, fake_settings)
    formatter = mod.CustomJsonFormatter("%(timestamp)s %(service_name)s %(message)s")

    logger, stream = _build_stream_logger("prediction", formatter)
    logger.warning("has ts", extra={"timestamp": "preserved-ts"})
    output = stream.getvalue().strip()
    data = json.loads(output)
    # Should include the provided timestamp rather than overriding
    assert data["timestamp"] == "preserved-ts"
    assert data["service_name"] == "press fault detection"
    assert data["message"] == "has ts"

def test_setup_loggers_creates_log_dir_and_configures_handlers(monkeypatch, fake_settings):
    mod = import_logger_module(monkeypatch, fake_settings)

    # Ensure clean loggers and non-existent handlers
    _clear_logger("system")
    _clear_logger("prediction")

    # Use controlled stdout so StreamHandlers bind to it
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout, raising=True)

    # Re-run setup with our patched settings
    mod.setup_loggers()

    # Verify directory creation
    assert os.path.isdir(fake_settings.LOG_DIR), "LOG_DIR should be created by setup_loggers"

    # Validate system logger configuration
    system_logger = logging.getLogger("system")
    assert system_logger.level == getattr(logging, fake_settings.LOG_LEVEL.upper())
    assert system_logger.propagate is False
    assert len(system_logger.handlers) >= 1
    sys_stream_handlers = [h for h in system_logger.handlers if isinstance(h, logging.StreamHandler)]
    assert sys_stream_handlers, "System logger should have a StreamHandler"
    # Emit a message and ensure it goes to our fake stdout
    system_logger.info("system configured")
    assert "system configured" in fake_stdout.getvalue()

    # Validate prediction logger configuration
    prediction_logger = logging.getLogger("prediction")
    assert prediction_logger.level == getattr(logging, fake_settings.LOG_LEVEL.upper())
    assert prediction_logger.propagate is False
    assert len(prediction_logger.handlers) >= 2, "Prediction logger should have console and file handlers"

    # One handler should be StreamHandler with LevelBasedFormatter
    stream_handlers = [h for h in prediction_logger.handlers if isinstance(h, logging.StreamHandler)]
    assert stream_handlers, "Prediction logger should have a StreamHandler"
    assert any(isinstance(h.formatter, mod.LevelBasedFormatter) for h in stream_handlers)

    # One handler should be TimedRotatingFileHandler with CustomJsonFormatter and WARNING level
    from logging.handlers import TimedRotatingFileHandler
    file_handlers = [h for h in prediction_logger.handlers if isinstance(h, TimedRotatingFileHandler)]
    assert file_handlers, "Prediction logger should have a TimedRotatingFileHandler"
    fh = file_handlers[0]
    assert isinstance(fh.formatter, mod.CustomJsonFormatter)
    assert fh.level == logging.WARNING

def test_prediction_file_handler_writes_only_warning_and_above(monkeypatch, fake_settings):
    mod = import_logger_module(monkeypatch, fake_settings)

    # Controlled stdout for console output
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout, raising=True)

    # Ensure clean and setup
    _clear_logger("prediction")
    mod.setup_loggers()
    prediction_logger = logging.getLogger("prediction")

    # Log INFO (should not be in file)
    prediction_logger.info("info should not be in file")

    # Log WARNING and ERROR (should be in file)
    prediction_logger.warning("warning should be in file")
    prediction_logger.error("error should be in file")

    # Flush file handlers
    for h in prediction_logger.handlers:
        with contextlib.suppress(Exception):
            h.flush()

    # Read and parse the file (json lines)
    assert os.path.isfile(fake_settings.ERROR_LOG_FILE_PATH), "Error log file must be created"
    with open(fake_settings.ERROR_LOG_FILE_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f.read().splitlines() if line.strip()]

    messages = [entry.get("message") for entry in lines]
    assert "warning should be in file" in messages
    assert "error should be in file" in messages
    assert "info should not be in file" not in messages
    # Also ensure required json fields exist
    for rec in lines:
        assert rec.get("service_name") == "press fault detection"
        assert isinstance(rec.get("timestamp"), str)

def test_system_logger_message_format_includes_timestamp_and_level(monkeypatch, fake_settings):
    mod = import_logger_module(monkeypatch, fake_settings)
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout, raising=True)

    _clear_logger("system")
    mod.setup_loggers()
    system_logger = logging.getLogger("system")
    system_logger.info("hello")

    out = fake_stdout.getvalue()
    # Expected format: "[YYYY-MM-DD HH:MM:SS] [LEVEL] message"
    assert "[" in out and "]" in out
    assert "[INFO]" in out
    assert "hello" in out