"""
FightMind AI — Cross-Encoder Re-Ranker
=======================================
Step 1.20 — Accuracy Refinement

After ChromaDB returns the initial bi-encoder candidates, this module
uses a Cross-Encoder to more precisely score each (query, chunk) pair.

Why Two Stages?
  - Bi-encoder (ChromaDB): Embeds query and chunks independently and compares
    vectors. Fast, but ignores token-level interaction between query and chunk.
  - Cross-encoder: Processes (query + chunk) as a single sequence, giving
    the model direct attention between every token. Much more accurate,
    but too slow to run over millions of docs — hence we use it only to
    re-rank the small candidate pool from ChromaDB.

This combination ("retrieve then re-rank") is the standard production
pattern for high-accuracy RAG systems.

Model Used:
    cross-encoder/ms-marco-MiniLM-L-6-v2
    - 85 MB, CPU-friendly, ~6ms per pair on a modern CPU.
    - Trained on Microsoft MARCO (1M question–passage pairs).
    - Top-ranked model on the BEIR retrieval benchmark.
"""

from typing import List, Dict, Any
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# The cross-encoder model is loaded lazily to avoid slowing down the FastAPI
# startup. The model is cached after the first call.
_cross_encoder = None
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    """Lazily load and cache the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model", extra={"model": CROSS_ENCODER_MODEL})
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder model loaded successfully")
    return _cross_encoder


def rerank(query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Re-ranks a list of ChromaDB search result dicts using a Cross-Encoder.

    Args:
        query:      The original user search query.
        candidates: List of result dicts from vector_store.search(), each
                    having at minimum a 'text' and 'score' key.
        top_n:      Number of results to return after re-ranking.

    Returns:
        A list of up to `top_n` result dicts, sorted by cross-encoder
        relevance score (highest first). Each result dict gets a new
        'rerank_score' key added for observability.
    """
    if not candidates:
        return []

    if len(candidates) <= top_n:
        # Not enough candidates to re-rank meaningfully — just return them all
        logger.debug(
            "Skipping re-rank (insufficient candidates)",
            extra={"candidates": len(candidates), "top_n": top_n},
        )
        return candidates

    try:
        cross_encoder = _get_cross_encoder()

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c["text"]) for c in candidates]

        # Predict returns one logit per pair; higher = more relevant
        scores = cross_encoder.predict(pairs)

        # Attach the rerank_score to each candidate for full traceability
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        # Sort by cross-encoder score descending, take top_n
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

        logger.info(
            "Re-ranking complete",
            extra={
                "query": query[:60],
                "input_candidates": len(candidates),
                "output_top_n": len(reranked),
                "top_score": round(reranked[0]["rerank_score"], 3) if reranked else None,
            },
        )
        return reranked

    except Exception as exc:
        logger.error("Cross-encoder re-ranking failed — returning original order", exc_info=exc)
        # Fail-open: return top_n from the original bi-encoder order
        return candidates[:top_n]
