"""
FightMind AI — Test Configuration
====================================
pytest conftest.py — shared fixtures for all tests.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient — used by all API tests."""
    with TestClient(app) as c:
        yield c
