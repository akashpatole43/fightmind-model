"""
Offline tests for Level 1 Intent Detection (Step 1.10).

Mocks the google.genai.Client to return deterministic Pydantic objects,
testing the logic without hitting the real API or needing a key.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.pipeline.level1_intent import (
    IntentCategory,
    IntentResult,
    SportType,
    detect_intent,
)


@pytest.fixture
def mock_genai_client():
    """Mocks the GenAI client and its generate_content response."""
    with patch("src.pipeline.level1_intent.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        mock_response = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        
        yield mock_client, mock_response


class TestDetectIntent:

    def test_empty_text_returns_generic(self):
        """Empty texts shouldn't call the API."""
        result = detect_intent("   \n ")
        assert result.category == IntentCategory.GENERIC_CHAT
        assert result.confidence == 1.0
        assert result.sport == SportType.UNKNOWN

    def test_detect_intent_success(self, mock_genai_client):
        """Happy path — tests SDK successfully parsing JSON into the model."""
        mock_client, mock_response = mock_genai_client
        
        # Simulate gemini returning a parsed Pydantic object
        mock_response.parsed = IntentResult(
            category=IntentCategory.LIVE_EVENT,
            sport=SportType.BOXING,
            confidence=0.95,
            extracted_entities=["Tyson Fury", "Usyk"]
        )
        
        result = detect_intent("when is fury vs usyk?")
        
        assert result.category == IntentCategory.LIVE_EVENT
        assert result.sport == SportType.BOXING
        assert result.extracted_entities == ["Tyson Fury", "Usyk"]
        
        # Verify the client was called with the right config
        mock_client.models.generate_content.assert_called_once()
        kwargs = mock_client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-1.5-flash"
        assert kwargs["contents"] == "when is fury vs usyk?"
        assert kwargs["config"].response_schema == IntentResult
        assert kwargs["config"].temperature == 0.0

    def test_api_failure_fallback(self, mock_genai_client):
        """Test that if the API fails, it safely falls back to GENERIC_CHAT."""
        mock_client, _ = mock_genai_client
        mock_client.models.generate_content.side_effect = Exception("API down")
        
        result = detect_intent("how to jab")
        
        assert result.category == IntentCategory.GENERIC_CHAT
        assert result.confidence == 0.0
