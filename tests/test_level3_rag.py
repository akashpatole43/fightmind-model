"""
Offline unit tests for Level 3 RAG Retrieval (Step 1.12).

Mocks the `src.rag.vector_store.search` function to test business logic
(query augmentation, domain guarding, filter construction) without
needing the real models or ChromaDB on disk.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.level1_intent import IntentCategory, IntentResult, SportType
from src.pipeline.level2_vision import VisionResult
from src.pipeline.level3_rag import retrieve_context, RagResult


@pytest.fixture
def mock_search():
    """Mocks the core vector store search function."""
    with patch("src.pipeline.level3_rag.search") as mock:
        yield mock


class TestRetrieveContext:

    def test_domain_guard_skips_search(self, mock_search):
        """If L1 says it's generic chat or out of domain, do not hit ChromaDB."""
        intent = IntentResult(
            category=IntentCategory.GENERIC_CHAT,
            sport=SportType.UNKNOWN,
            confidence=0.9,
            extracted_entities=[]
        )
        
        result = retrieve_context("hello", intent_result=intent)
        
        assert mock_search.call_count == 0
        assert result.max_score == 0.0
        assert len(result.retrieved_chunks) == 0

    def test_standard_text_search(self, mock_search):
        """Happy path — text query is mapped directly to a search with sport filters."""
        # Setup mock db return (uses the correct `text` and `score` keys)
        mock_search.return_value = [
            {"text": "A jab is...", "score": 0.85},
            {"text": "Keep hands up...", "score": 0.60}
        ]
        
        intent = IntentResult(
            category=IntentCategory.TECHNIQUE,
            sport=SportType.BOXING,
            confidence=0.9,
            extracted_entities=[]
        )
        
        result = retrieve_context("how to jab", intent_result=intent)
        
        # Verify Search arguments use the new Step 1.20 reranking signature
        mock_search.assert_called_once_with(
            query="how to jab",
            top_k=10,
            sport="boxing",
            rerank=True,
            rerank_top_n=3,
        )
        
        # Verify output parsing uses `text` key
        assert result.max_score == 0.85
        assert len(result.retrieved_chunks) == 2
        assert result.retrieved_chunks[0] == "A jab is..."

    def test_vision_augmentation(self, mock_search):
        """If an image was uploaded, L2's text should be appended to the RAG query."""
        mock_search.return_value = [{"text": "Roundhouse...", "score": 0.9}]
        
        intent = IntentResult(
            category=IntentCategory.TECHNIQUE,
            sport=SportType.UNKNOWN,
            confidence=0.5,
            extracted_entities=[]
        )
        
        vision = VisionResult(
            description="A high kick.",
            extracted_techniques=["Roundhouse Kick"],
            confidence=0.95
        )
        
        result = retrieve_context("what is this?", intent_result=intent, vision_result=vision)
        
        # Verify the query was rewritten to include the image context
        mock_search.assert_called_once_with(
            query="what is this? [Image Context: A high kick.] [Techniques: Roundhouse Kick]",
            top_k=10,
            sport=None,   # UNKNOWN sport shouldn't pass a filter
            rerank=True,
            rerank_top_n=3,
        )
        
        assert result.max_score == 0.9

    def test_search_failure_graceful_fallback(self, mock_search):
        """If ChromaDB throws an exception, handle it and flag for fallback."""
        mock_search.side_effect = Exception("ChromaDB is offline")
        
        intent = IntentResult(
            category=IntentCategory.TECHNIQUE,
            sport=SportType.BOXING,
            confidence=0.9,
            extracted_entities=[]
        )
        
        result = retrieve_context("how to jab", intent_result=intent)
        
        assert result.used_fallback is True
        assert result.max_score == 0.0
        assert len(result.retrieved_chunks) == 0
