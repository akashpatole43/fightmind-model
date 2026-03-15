"""
FightMind AI — RAG Retrieval Evaluation
=======================================
Step 1.9 — Benchmarks the vector store's retrieval accuracy.

Runs a predefined set of Queries and checks if the expected Wikipedia
document appears in the Top-K results.

Metrics calculated:
  - Hit@1: % of queries where the correct doc is the #1 result
  - Hit@3: % of queries where the correct doc is in the top 3
  - Hit@5: % of queries where the correct doc is in the top 5
  - MRR:   Mean Reciprocal Rank (1/rank of first correct result)
"""

from typing import Optional

from src.core.logging_config import get_logger
from src.rag.vector_store import search

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Test Dataset
# ─────────────────────────────────────────────────────────────────────────────

# Format: ("Query text", "Expected Document Title")
EVAL_QUERIES: list[tuple[str, str]] = [
    # Boxing
    ("how do you throw a jab?",           "Jab (boxing)"),
    ("what is a liver shot?",             "Liver shot"),
    ("what are the rules of boxing?",     "WBC Rules"),
    ("what is the southpaw stance?",      "Southpaw stance"),
    ("how many points for a knockdown?",  "10-Point Must System"),

    # Kickboxing
    ("how to execute a low kick",         "Low kick"),
    ("what is a roundhouse kick?",        "Roundhouse kick"),
    ("are elbows allowed in k-1?",        "K-1 Rules"),
    ("what is muay thai clinch?",         "Clinch fighting"),

    # Karate
    ("what is ippon in kumite?",          "Ippon"),
    ("how to do a front kick in karate",  "Mae geri"),
    ("what does sensei mean?",            "Sensei"),
    ("what is a kata in shotokan?",       "Kata"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_retrieval(
    queries: list[tuple[str, str]] = EVAL_QUERIES,
    top_k: int = 5
) -> dict[str, float]:
    """
    Run evaluation queries against the vector store and calculate metrics.

    Args:
        queries: List of (query_text, expected_doc_title)
        top_k:   Maximum results to fetch per query (defines Hit@K bounds)

    Returns:
        Dictionary of metrics: {"hit@1": 0.X, "hit@3": 0.X, "hit@5": 0.X, "mrr": 0.X}
    """
    if not queries:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}

    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    mrr_sum = 0.0

    logger.info("Starting RAG evaluation", extra={"queries": len(queries), "top_k": top_k})

    for query, expected_title in queries:
        # Search vector store (all sports)
        results = search(query, top_k=top_k)

        # Find the rank (1-indexed) of the first result matching the expected title
        rank: Optional[int] = None
        for i, res in enumerate(results):
            # Case-insensitive title match for robustness
            if expected_title.lower() in res["metadata"]["doc_title"].lower():
                rank = i + 1
                break

        # Calculate metrics for this query
        if rank is not None:
            if rank == 1:
                hits_1 += 1
            if rank <= 3:
                hits_3 += 1
            if rank <= 5:
                hits_5 += 1

            mrr_sum += 1.0 / rank
            logger.debug(
                "Query Hit",
                extra={"query": query, "expected": expected_title, "rank": rank}
            )
        else:
            logger.warning(
                "Query Miss (not in top_k)",
                extra={"query": query, "expected": expected_title, "top_k": top_k}
            )

    # Aggregate
    num_queries = len(queries)
    metrics = {
        "hit@1": round(hits_1 / num_queries, 4),
        "hit@3": round(hits_3 / num_queries, 4),
        "hit@5": round(hits_5 / num_queries, 4),
        "mrr":   round(mrr_sum / num_queries, 4),
    }

    logger.info("Evaluation complete", extra=metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    
    # We only want INFO and above for the script output
    setup_logging()

    print("\n" + "="*50)
    print("FightMind AI — RAG Evaluation")
    print("="*50)

    metrics = evaluate_retrieval()

    print("\n📊 Results:")
    print(f"  Hit@1 : {metrics['hit@1']*100:6.2f}%")
    print(f"  Hit@3 : {metrics['hit@3']*100:6.2f}%")
    print(f"  Hit@5 : {metrics['hit@5']*100:6.2f}%")
    print(f"  MRR   : {metrics['mrr']:6.4f}")
    
    if metrics["mrr"] > 0.6:
        print("\n✅ Status: PASS (MRR > 0.6)")
    else:
        print("\n❌ Status: FAIL (MRR <= 0.6 — embeddings need improvement)")
    print("="*50 + "\n")
