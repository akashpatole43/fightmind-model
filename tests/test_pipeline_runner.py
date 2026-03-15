"""
Offline unit tests for the Pipeline Orchestrator (Step 1.16).
Verifies that `run_pipeline` calls the inner modules correctly.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.pipeline_runner import run_pipeline, ChatbotResponse
from src.pipeline.level1_intent import IntentCategory, IntentResult, SportType
from src.pipeline.level2_vision import VisionResult
from src.pipeline.level3_rag import RagResult
from src.pipeline.level4_events import EventsResult
from src.pipeline.level5_llm import LlmResult
from src.pipeline.level6_validate import ValidationResult, SkillLevel


@pytest.fixture
def mock_pipeline_levels():
    """Mocks all 6 levels to run offline integration tests."""
    with patch("src.pipeline_runner.detect_intent") as m_l1, \
         patch("src.pipeline_runner.analyze_image") as m_l2, \
         patch("src.pipeline_runner.retrieve_context") as m_l3, \
         patch("src.pipeline_runner.fetch_live_context") as m_l4, \
         patch("src.pipeline_runner.generate_answer") as m_l5, \
         patch("src.pipeline_runner.validate_answer") as m_l6:
         
         # Default happy-path mock returns
         m_l1.return_value = IntentResult(category=IntentCategory.TECHNIQUE, sport=SportType.BOXING, confidence=0.9, extracted_entities=[])
         m_l2.return_value = VisionResult(description="mock image", extracted_techniques=[], confidence=0.9)
         m_l3.return_value = RagResult(retrieved_chunks=["mock chunk"], max_score=0.9, used_fallback=False)
         m_l4.return_value = EventsResult(event_context="mock events", has_events=False, used_fallback=False)
         m_l5.return_value = LlmResult(answer="mock final answer", confidence=0.95, used_fallback=False)
         m_l6.return_value = ValidationResult(is_safe=True, hallucination_detected=False, detected_skill_level=SkillLevel.BEGINNER)
         
         yield (m_l1, m_l2, m_l3, m_l4, m_l5, m_l6)


class TestPipelineRunner:

    def test_full_successful_execution(self, mock_pipeline_levels):
        """Verifies all 6 levels are called for a standard technique/image query."""
        m_l1, m_l2, m_l3, m_l4, m_l5, m_l6 = mock_pipeline_levels
        
        res = run_pipeline("mock query", image_url="http://mock.com")
        
        m_l1.assert_called_once()
        m_l2.assert_called_once()
        m_l3.assert_called_once()
        m_l4.assert_called_once()
        m_l5.assert_called_once()
        m_l6.assert_called_once()
        
        assert isinstance(res, ChatbotResponse)
        assert res.answer == "mock final answer"
        assert res.user_skill_level == "BEGINNER"
        assert res.used_vision is True

    def test_skips_l2_if_no_image(self, mock_pipeline_levels):
        """If no image URL is provided, Level 2 should be skipped."""
        m_l1, m_l2, m_l3, m_l4, m_l5, m_l6 = mock_pipeline_levels
        
        res = run_pipeline("mock query", image_url=None)
        
        m_l2.assert_not_called()
        assert res.used_vision is False

    def test_short_circuits_out_of_domain(self, mock_pipeline_levels):
        """If L1 flags as out-of-domain, L2-L6 must be skipped entirely."""
        m_l1, m_l2, m_l3, m_l4, m_l5, m_l6 = mock_pipeline_levels
        
        m_l1.return_value = IntentResult(category=IntentCategory.OUT_OF_DOMAIN, sport=SportType.UNKNOWN, confidence=0.9, extracted_entities=[])
        
        res = run_pipeline("mock out of domain query")
        
        m_l1.assert_called_once()
        m_l2.assert_not_called()
        m_l3.assert_not_called()
        m_l5.assert_not_called()
        
        assert "I am FightMind AI" in res.answer
        assert res.detected_intent == "OUT_OF_DOMAIN"

    def test_safety_check_overrides_answer(self, mock_pipeline_levels):
        """If L6 flags an answer as unsafe, it should overwrite the L5 response."""
        m_l1, m_l2, m_l3, m_l4, m_l5, m_l6 = mock_pipeline_levels
        
        m_l6.return_value = ValidationResult(is_safe=False, hallucination_detected=False, detected_skill_level=SkillLevel.UNKNOWN)
        
        res = run_pipeline("how to street fight")
        
        assert "This query touches upon dangerous, illegal, or unsanctioned" in res.answer
        assert res.hallucination_flag is False

    def test_appends_hallucination_disclaimer(self, mock_pipeline_levels):
        """If L6 flags a hallucination, a disclaimer should be appended."""
        m_l1, m_l2, m_l3, m_l4, m_l5, m_l6 = mock_pipeline_levels
        
        m_l6.return_value = ValidationResult(is_safe=True, hallucination_detected=True, detected_skill_level=SkillLevel.ADVANCED)
        
        res = run_pipeline("mock query")
        
        assert res.answer.startswith("mock final answer")
        assert "Disclaimer:" in res.answer
        assert res.hallucination_flag is True
