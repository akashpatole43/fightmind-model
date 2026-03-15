"""
Offline unit tests for Level 5 LLM Generation (Step 1.14).
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.pipeline.level1_intent import IntentCategory, IntentResult
from src.pipeline.level2_vision import VisionResult
from src.pipeline.level3_rag import RagResult
from src.pipeline.level4_events import EventsResult
from src.pipeline.level5_llm import generate_answer, LlmResult, _build_prompt


class _MockParsedResponse(BaseModel):
    parsed: LlmResult


@pytest.fixture
def mock_gemini():
    """Mocks the whole google.genai.Client to prevent API calls."""
    with patch("src.pipeline.level5_llm.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Create a mock response object that matches structured output behavior
        mock_response = MagicMock()
        mock_response.parsed = LlmResult(
            answer="This is a mocked correct answer from the LLM.",
            confidence=0.95,
            used_fallback=False
        )
        
        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response
        mock_client.models = mock_models
        
        yield mock_client


class TestGenerateAnswer:

    def test_skips_out_of_domain(self, mock_gemini):
        """Should return a canned response without calling the API if L1 flagged it."""
        intent = IntentResult(
            category=IntentCategory.OUT_OF_DOMAIN,
            sport="UNKNOWN",
            confidence=0.9,
            extracted_entities=[]
        )
        
        result = generate_answer("How do I bake a cake?", intent)
        
        assert mock_gemini.models.generate_content.call_count == 0
        assert "I am FightMind AI" in result.answer
        assert result.confidence == 1.0
        assert result.used_fallback is False

    def test_compiles_prompt_and_calls_api(self, mock_gemini):
        """Should compile all provided contexts and call Gemini."""
        intent = IntentResult(
            category=IntentCategory.TECHNIQUE,
            sport="BOXING",
            confidence=0.9,
            extracted_entities=[]
        )
        rag = RagResult(
            retrieved_chunks=["Keep hands up."],
            max_score=0.9,
            used_fallback=False
        )
        
        result = generate_answer("How to box?", intent, rag_res=rag)
        
        mock_gemini.models.generate_content.assert_called_once()
        
        # Verify the prompt string passed into generate_content contained our context
        call_args = mock_gemini.models.generate_content.call_args[1]
        compiled_string = call_args["contents"]
        
        assert "How to box?" in compiled_string
        assert "Keep hands up." in compiled_string
        assert "[KNOWLEDGE BASE CONTEXT]" in compiled_string
        assert result.answer == "This is a mocked correct answer from the LLM."

    def test_handles_api_failure_fallback(self, mock_gemini):
        """Should return a safe fallback message if Gemini throws an error."""
        mock_gemini.models.generate_content.side_effect = Exception("API Quota Reached")
        
        intent = IntentResult(
            category=IntentCategory.TECHNIQUE,
            sport="BOXING",
            confidence=0.9,
            extracted_entities=[]
        )
        
        result = generate_answer("test", intent)
        
        assert result.used_fallback is True
        assert result.confidence == 0.0
        assert "apologize" in result.answer

class TestPromptBuilder:
    def test_build_prompt_all_components(self):
        """Tests that the string builder correctly concats all the pieces."""
        intent = IntentResult(
            category=IntentCategory.LIVE_EVENT,
            sport="UNKNOWN",
            confidence=0.9,
            extracted_entities=[]
        )
        vision = VisionResult(
            description="A guy in red shorts kicking.",
            extracted_techniques=["roundhouse kick"],
            confidence=0.8
        )
        rag = RagResult(retrieved_chunks=["Chunk 1", "Chunk 2"], max_score=0.9, used_fallback=False)
        events = EventsResult(event_context="UFC 300 is tonight.", has_events=True, used_fallback=False)
        
        prompt = _build_prompt("When is the fight and what kick is this?", intent, vision, rag, events)
        
        assert "USER QUERY:\nWhen is the fight" in prompt
        assert "[USER UPLOADED IMAGE]" in prompt
        assert "A guy in red shorts" in prompt
        assert "roundhouse kick" in prompt
        assert "[KNOWLEDGE BASE CONTEXT]" in prompt
        assert "Chunk 1" in prompt
        assert "[LIVE EVENTS CONTEXT]" in prompt
        assert "UFC 300 is tonight." in prompt
