"""
Offline unit tests for Level 6 Validate & Personalize (Step 1.15).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.pipeline.level3_rag import RagResult
from src.pipeline.level6_validate import validate_answer, ValidationResult, SkillLevel, _GeminiValidationResult


class _MockParsedResponse(BaseModel):
    parsed: _GeminiValidationResult


@pytest.fixture
def mock_gemini():
    """Mocks the whole google.genai.Client to prevent API calls."""
    with patch("src.pipeline.level6_validate.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Default mock response object
        mock_response = MagicMock()
        mock_response.parsed = _GeminiValidationResult(
            is_safe=True,
            hallucination_detected=False,
            detected_skill_level="BEGINNER"
        )
        
        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response
        mock_client.models = mock_models
        
        yield mock_client


class TestValidateAnswer:

    def test_validates_safe_accurate_answer(self, mock_gemini):
        """Should return the mapped enums and exact bools from the mock SDK."""
        rag = RagResult(retrieved_chunks=["Keep hands up."], max_score=0.9, used_fallback=False)
        result = validate_answer("How to box?", rag, "Keep your hands up.")
        
        mock_gemini.models.generate_content.assert_called_once()
        
        assert result.is_safe is True
        assert result.hallucination_detected is False
        assert result.detected_skill_level == SkillLevel.BEGINNER

    def test_maps_incorrect_enum_to_unknown(self, mock_gemini):
        """If Gemini SDK returns 'PRO_FIGHTER', we should gracefully map it to UNKNOWN."""
        
        # Override mock specific for this test
        mock_response = MagicMock()
        mock_response.parsed = _GeminiValidationResult(
            is_safe=True,
            hallucination_detected=False,
            detected_skill_level="PRO_FIGHTER" # INVALID
        )
        mock_gemini.models.generate_content.return_value = mock_response
        
        rag = RagResult(retrieved_chunks=[], max_score=0.0, used_fallback=False)
        result = validate_answer("fake", rag, "fake")
        
        assert result.detected_skill_level == SkillLevel.UNKNOWN

    def test_api_failure_fails_open(self, mock_gemini):
        """If the Google API is down, validation should FAIL OPEN to let the message through."""
        mock_gemini.models.generate_content.side_effect = Exception("500 Server Error")
        
        rag = RagResult(retrieved_chunks=[], max_score=0.0, used_fallback=False)
        result = validate_answer("fake", rag, "fake")
        
        assert result.is_safe is True
        assert result.hallucination_detected is False
        assert result.detected_skill_level == SkillLevel.UNKNOWN
