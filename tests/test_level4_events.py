"""
Offline unit tests for Level 4 Live Events (Step 1.13).

Mocks the `src.data_collection.sports_api` client to verify that the
data formatting and intent-based short-circuiting work correctly 
without hitting TheSportsDB.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.level1_intent import IntentCategory, IntentResult
from src.pipeline.level4_events import fetch_live_context, EventsResult


@pytest.fixture
def mock_sports_api():
    """Mocks the external sports API call."""
    with patch("src.pipeline.level4_events.fetch_events") as mock_fetch:
        yield mock_fetch


class TestFetchLiveContext:

    def test_skips_if_not_live_event_intent(self, mock_sports_api):
        """If L1 says it's a technique, do not waste an API call."""
        intent = IntentResult(
            category=IntentCategory.TECHNIQUE,
            sport="BOXING",
            confidence=0.9,
            extracted_entities=["UFC"]  # Even if UFC is found, category rules
        )
        
        result = fetch_live_context(intent)
        
        assert mock_sports_api.call_count == 0
        assert result.has_events is False
        assert result.event_context == ""

    def test_fetches_ufc_default(self, mock_sports_api):
        """If intent is LIVE_EVENT and entity is UFC, allow all events through filter."""
        # Mock API returning JSON data matching Event dataclass format
        mock_sports_api.return_value = [
            {
                "title": "UFC 300: Pereira vs Hill",
                "sport": "mixed martial arts",
                "date": "2024-04-13",
                "time": "22:00:00",
                "status": "Scheduled",
                "venue": "T-Mobile Arena",
                "promotion": "UFC",
                "fighters": "Alex Pereira vs Jamahal Hill"
            }
        ]
        
        intent = IntentResult(
            category=IntentCategory.LIVE_EVENT,
            sport="UNKNOWN",
            confidence=0.9,
            extracted_entities=["UFC"]
        )
        
        result = fetch_live_context(intent)
        
        mock_sports_api.assert_called_once()
        
        assert result.has_events is True
        assert "UFC 300" in result.event_context
        assert "2024-04-13" in result.event_context

    def test_fetches_specific_entity_filter(self, mock_sports_api):
        """If intent is LIVE_EVENT with a specific name, filter the returned events."""
        mock_sports_api.return_value = [
            {
                "title": "UFC 301",
                "date": "2024-05-04",
            },
            {
                "title": "Canelo vs Munguia",
                "date": "2024-05-04",
                "fighters": "Canelo Alvarez vs Jaime Munguia"
            }
        ]
        
        intent = IntentResult(
            category=IntentCategory.LIVE_EVENT,
            sport="BOXING",
            confidence=0.9,
            extracted_entities=["Canelo"]
        )
        
        result = fetch_live_context(intent)
        
        mock_sports_api.assert_called_once()
        
        assert result.has_events is True
        assert "Canelo vs Munguia" in result.event_context
        assert "UFC 301" not in result.event_context  # Filtered out!

    def test_api_failure_fallback(self, mock_sports_api):
        """If the external API crashes, handle it gracefully so the bot doesn't crash."""
        mock_sports_api.side_effect = Exception("TheSportsDB API offline")
        
        intent = IntentResult(
            category=IntentCategory.LIVE_EVENT,
            sport="UNKNOWN",
            confidence=0.9,
            extracted_entities=["UFC"]
        )
        
        result = fetch_live_context(intent)
        
        assert result.used_fallback is True
        assert result.has_events is False
        assert "unavailable" in result.event_context
