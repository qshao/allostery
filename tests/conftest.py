from pathlib import Path

import pytest


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures"
