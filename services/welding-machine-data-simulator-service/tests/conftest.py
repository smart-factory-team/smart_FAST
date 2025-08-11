import pytest
import asyncio
import sys
import os

# Configure pytest for async testing
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Add project root to Python path - following project pattern
@pytest.fixture(autouse=True)
def setup_python_path():
    """Automatically add project paths for all tests."""
    project_root = os.path.join(os.path.dirname(__file__), '../')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

@pytest.fixture
def mock_print(monkeypatch):
    """Mock print function to capture output in tests."""
    outputs = []
    def mock_print_func(*args, **kwargs):
        outputs.append(' '.join(str(arg) for arg in args))
    
    monkeypatch.setattr('builtins.print', mock_print_func)
    return outputs