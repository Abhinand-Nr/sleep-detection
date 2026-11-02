from pathlib import Path

import pytest

from src.utils.config import load_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg() -> dict:
    return load_config(ROOT / "config.yaml")
