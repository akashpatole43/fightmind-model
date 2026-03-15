"""
Offline tests for the RAG evaluation script (Step 1.9).

Ensures the Hit@k and MRR metric calculations are correct by mocking
the vector store search response.
"""

from unittest.mock import patch

from src.training.evaluate import evaluate_retrieval


class TestRagEvaluation:
    @patch("src.training.evaluate.search")
    def test_evaluate_retrieval_perfect_score(self, mock_search):
        """Test metrics when every query returns the correct doc at rank 1."""
        queries = [
            ("q1", "docA"),
            ("q2", "docB"),
        ]
        
        # Mock search to always return the expected doc at index 0
        def mock_search_perfect(query, top_k):
            # Find the expected title for this query from the test queries list
            expected = next((title for q, title in queries if q == query), "unknown")
            return [
                {"metadata": {"doc_title": expected}},
                {"metadata": {"doc_title": "wrong"}},
            ]
            
        mock_search.side_effect = mock_search_perfect
        
        metrics = evaluate_retrieval(queries, top_k=5)
        
        assert metrics["hit@1"] == 1.0   # 100%
        assert metrics["hit@3"] == 1.0
        assert metrics["hit@5"] == 1.0
        assert metrics["mrr"] == 1.0

    @patch("src.training.evaluate.search")
    def test_evaluate_retrieval_mixed_ranks(self, mock_search):
        """Test MRR calculus with ranks 1, 2, and Miss."""
        queries = [
            ("q1", "docA"),  # Will rank 1
            ("q2", "docB"),  # Will rank 2
            ("q3", "docC"),  # Will miss (not in top k)
        ]
        
        # Hardcode the mock responses chronologically for the 3 queries
        mock_search.side_effect = [
            [{"metadata": {"doc_title": "docA"}}, {"metadata": {"doc_title": "wrong"}}],              # Rank 1
            [{"metadata": {"doc_title": "wrong"}}, {"metadata": {"doc_title": "docB"}}],              # Rank 2
            [{"metadata": {"doc_title": "wrong"}}, {"metadata": {"doc_title": "wrong_again"}}],       # Miss
        ]
        
        metrics = evaluate_retrieval(queries, top_k=2)
        
        # hit@1 = 1/3 (docA at rank 1)  -> 0.3333
        # hit@3 = 2/3 (docA, docB)      -> 0.6667
        # hit@5 = 2/3 (docA, docB)      -> 0.6667
        # mrr   = (1/1 + 1/2 + 0) / 3   -> 1.5 / 3 = 0.5
        
        assert metrics["hit@1"] == 0.3333
        assert metrics["hit@3"] == 0.6667
        assert metrics["hit@5"] == 0.6667
        assert metrics["mrr"] == 0.5000

    def test_evaluate_retrieval_empty_queries(self):
        """Test safe handling of empty query lists."""
        metrics = evaluate_retrieval([])
        assert metrics["hit@1"] == 0.0
        assert metrics["mrr"] == 0.0
