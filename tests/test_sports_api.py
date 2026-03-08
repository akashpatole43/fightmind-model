"""
Tests for src/data_collection/sports_api.py
=============================================
All external HTTP calls are mocked — no internet or API keys required.

Tests cover:
  - TheSportsDB happy path
  - API-Sports happy path
  - 3-source fallback logic (Source 1 empty → Source 2, etc.)
  - Graceful handling of network errors, malformed JSON, missing key
  - save_events file output
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Bootstrap logging before importing the module under test
from src.core.logging_config import setup_logging
setup_logging()

from src.data_collection.sports_api import (
    Event,
    fetch_events,
    save_events,
    _fetch_thesportsdb_events,
    _fetch_api_sports_events,
    _fetch_scraped_events,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

TEST_DATE = "2026-03-08"

THESPORTSDB_RESPONSE = {
    "events": [
        {
            "strEvent":  "Fury vs Usyk 2",
            "dateEvent": "2026-03-08",
            "strTime":   "20:00:00",
            "strVenue":  "Kingdom Arena, Riyadh",
            "strLeague": "WBC",
            "strStatus": "scheduled",
        },
    ]
}

API_SPORTS_RESPONSE = {
    "response": [
        {
            "date": "2026-03-08",
            "time": "21:00",
            "boxer1": {"name": "Canelo Álvarez"},
            "boxer2": {"name": "Jermall Charlo"},
            "location": {"venue": "T-Mobile Arena"},
            "promotion": {"name": "Golden Boy"},
            "status": "scheduled",
        }
    ]
}


def _mock_response(json_body: dict, status_code: int = 200) -> MagicMock:
    """Build a fake requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_body
    mock.raise_for_status.return_value = None
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# TheSportsDB Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTheSportsDB:

    @patch("src.data_collection.sports_api._get_with_retry")
    def test_returns_events_on_success(self, mock_get):
        """Happy path — API returns events."""
        mock_get.return_value = _mock_response(THESPORTSDB_RESPONSE)

        events = _fetch_thesportsdb_events(TEST_DATE)

        assert len(events) >= 1
        assert isinstance(events[0], Event)
        assert events[0].sport in ("boxing", "kickboxing", "karate")
        assert events[0].source == "thesportsdb"

    @patch("src.data_collection.sports_api._get_with_retry")
    def test_empty_when_no_events_in_response(self, mock_get):
        """API returns null events array — should return empty list, not crash."""
        mock_get.return_value = _mock_response({"events": None})

        events = _fetch_thesportsdb_events(TEST_DATE)
        assert events == []

    @patch("src.data_collection.sports_api._get_with_retry")
    def test_empty_when_request_fails(self, mock_get):
        """Network failure — should return empty list, not raise."""
        mock_get.return_value = None   # simulates all retries exhausted

        events = _fetch_thesportsdb_events(TEST_DATE)
        assert events == []

    @patch("src.data_collection.sports_api._get_with_retry")
    def test_skips_malformed_json(self, mock_get):
        """API returns unparseable body — should return empty list, not raise."""
        bad_resp = MagicMock()
        bad_resp.raise_for_status.return_value = None
        bad_resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)
        mock_get.return_value = bad_resp

        events = _fetch_thesportsdb_events(TEST_DATE)
        assert events == []


# ─────────────────────────────────────────────────────────────────────────────
# API-Sports Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAPISports:

    @patch("src.data_collection.sports_api.API_SPORTS_KEY", "test-key-123")
    @patch("src.data_collection.sports_api._get_with_retry")
    def test_returns_events_on_success(self, mock_get):
        """Happy path — API returns fight data."""
        mock_get.return_value = _mock_response(API_SPORTS_RESPONSE)

        events = _fetch_api_sports_events(TEST_DATE)

        assert len(events) == 1
        assert events[0].sport == "boxing"
        assert "Canelo" in events[0].fighters
        assert events[0].source == "api_sports"

    @patch("src.data_collection.sports_api.API_SPORTS_KEY", "")
    def test_returns_empty_when_no_key(self):
        """No API key configured — skip gracefully."""
        events = _fetch_api_sports_events(TEST_DATE)
        assert events == []

    @patch("src.data_collection.sports_api.API_SPORTS_KEY", "test-key")
    @patch("src.data_collection.sports_api._get_with_retry")
    def test_returns_empty_when_request_fails(self, mock_get):
        """Network failure — should return empty list."""
        mock_get.return_value = None
        events = _fetch_api_sports_events(TEST_DATE)
        assert events == []


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Orchestrator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchEventsOrchestrator:

    @patch("src.data_collection.sports_api._fetch_thesportsdb_events")
    def test_returns_source1_when_available(self, mock_s1):
        """Source 1 works — use it and skip Sources 2 & 3."""
        sample_event = Event(title="Fight A", sport="boxing", date=TEST_DATE, source="thesportsdb")
        mock_s1.return_value = [sample_event]

        result = fetch_events(TEST_DATE)

        assert len(result) == 1
        assert result[0]["source"] == "thesportsdb"
        mock_s1.assert_called_once()

    @patch("src.data_collection.sports_api._fetch_scraped_events")
    @patch("src.data_collection.sports_api._fetch_api_sports_events")
    @patch("src.data_collection.sports_api._fetch_thesportsdb_events")
    def test_falls_back_to_source2_when_source1_empty(self, mock_s1, mock_s2, mock_s3):
        """Source 1 empty → falls back to Source 2."""
        mock_s1.return_value = []
        sample_event = Event(title="Fight B", sport="boxing", date=TEST_DATE, source="api_sports")
        mock_s2.return_value = [sample_event]

        result = fetch_events(TEST_DATE)

        assert len(result) == 1
        assert result[0]["source"] == "api_sports"
        mock_s3.assert_not_called()

    @patch("src.data_collection.sports_api._fetch_scraped_events")
    @patch("src.data_collection.sports_api._fetch_api_sports_events")
    @patch("src.data_collection.sports_api._fetch_thesportsdb_events")
    def test_falls_back_to_source3_when_sources_1_2_empty(self, mock_s1, mock_s2, mock_s3):
        """Source 1 & 2 empty → falls back to Source 3 web scraping."""
        mock_s1.return_value = []
        mock_s2.return_value = []
        sample_event = Event(title="Fight C", sport="kickboxing", date=TEST_DATE, source="web_scrape")
        mock_s3.return_value = [sample_event]

        result = fetch_events(TEST_DATE)

        assert len(result) == 1
        assert result[0]["source"] == "web_scrape"

    @patch("src.data_collection.sports_api._fetch_scraped_events")
    @patch("src.data_collection.sports_api._fetch_api_sports_events")
    @patch("src.data_collection.sports_api._fetch_thesportsdb_events")
    def test_returns_empty_list_when_all_sources_fail(self, mock_s1, mock_s2, mock_s3):
        """All 3 sources empty — returns [] without raising."""
        mock_s1.return_value = []
        mock_s2.return_value = []
        mock_s3.return_value = []

        result = fetch_events(TEST_DATE)
        assert result == []

    @patch("src.data_collection.sports_api._fetch_scraped_events")
    @patch("src.data_collection.sports_api._fetch_api_sports_events")
    @patch("src.data_collection.sports_api._fetch_thesportsdb_events")
    def test_handles_exception_in_source1_gracefully(self, mock_s1, mock_s2, mock_s3):
        """Source 1 raises unexpectedly — should continue to Source 2."""
        mock_s1.side_effect = RuntimeError("unexpected crash")
        mock_s2.return_value = [Event(title="Fight D", sport="boxing", date=TEST_DATE, source="api_sports")]
        mock_s3.return_value = []

        result = fetch_events(TEST_DATE)
        assert len(result) == 1
        assert result[0]["source"] == "api_sports"


# ─────────────────────────────────────────────────────────────────────────────
# save_events Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveEvents:

    def test_saves_events_to_json(self, tmp_path):
        """Events list is serialized to a JSON file."""
        # Override RAW_DIR to the temp directory for this test
        import src.data_collection.sports_api as module
        original_raw_dir = module.RAW_DIR
        module.RAW_DIR = tmp_path

        try:
            events = [{"title": "Fight X", "sport": "boxing", "date": TEST_DATE}]
            path = save_events(events, filename="test_events.json")

            assert path.exists()
            loaded = json.loads(path.read_text(encoding="utf-8"))
            assert len(loaded) == 1
            assert loaded[0]["title"] == "Fight X"
        finally:
            module.RAW_DIR = original_raw_dir

    def test_raises_on_permission_error(self, tmp_path):
        """IOError during write is logged at CRITICAL and re-raised."""
        import src.data_collection.sports_api as module
        original_raw_dir = module.RAW_DIR
        module.RAW_DIR = tmp_path

        try:
            with patch("builtins.open", side_effect=PermissionError("no write access")):
                with pytest.raises(PermissionError):
                    save_events([{"title": "Fight Y"}], filename="fail.json")
        finally:
            module.RAW_DIR = original_raw_dir
