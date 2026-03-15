"""
FightMind AI — Level 3: RAG Retrieval
=====================================
Step 1.12 — Connects the Level 1 and Level 2 outputs to the ChromaDB
vector database we built in Step 1.8.

It takes the user's base query, appends any image descriptions,
filters by the detected sport, and retrieves the top-K relevant 
chunks of martial arts knowledge to feed to the final LLM.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

from src.core.logging_config import get_logger
from src.pipeline.level1_intent import IntentResult, IntentCategory, SportType
from src.pipeline.level2_vision import VisionResult
from src.rag.vector_store import search, get_collection

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Structured Output Schema (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class RagResult(BaseModel):
    retrieved_chunks: List[str] = Field(
        description="The relevant text blocks retrieved from the ChromaDB vector store."
    )
    max_score: float = Field(
        description="The highest similarity score amongst the retrieved chunks (0.0 to 1.0). If 0.0, nothing relevant was found."
    )
    used_fallback: bool = Field(
        description="True if the search failed or the collection was missing, meaning we are falling back to general LLM knowledge."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Implementation
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    intent_result: IntentResult,
    vision_result: Optional[VisionResult] = None,
    top_k: int = 3
) -> RagResult:
    """
    Searches the RAG ChromaDB database intelligently based on prior pipeline levels.

    Args:
        query: The user's original raw text prompt.
        intent_result: The L1 classification (decides filters & whether to search at all).
        vision_result: The L2 image analysis (appended to the query if present).
        top_k: Number of chunks to retrieve.

    Returns:
        RagResult containing the literal text chunks to pass to Level 5.
    """
    
    # 1. Domain Guard — Don't waste time searching for cake recipes
    if intent_result.category in (IntentCategory.OUT_OF_DOMAIN, IntentCategory.GENERIC_CHAT):
        logger.info("Skipping RAG retrieval", extra={"reason": intent_result.category.value})
        return RagResult(retrieved_chunks=[], max_score=0.0, used_fallback=False)
        
    # 2. Build the Hybrid Query
    # If the user uploaded an image of a liver hook and said "what is this",
    # "what is this" is useless to ChromaDB. We must append the Vision description.
    search_query = query.strip()
    
    if vision_result and vision_result.description:
        if search_query:
            search_query += " "
        search_query += f"[Image Context: {vision_result.description}]"
        
        # Also append identified techniques to heavily weight the embeddings
        if vision_result.extracted_techniques:
            techs = ", ".join(vision_result.extracted_techniques)
            search_query += f" [Techniques: {techs}]"

    if not search_query:
        logger.warning("Search query is completely empty after building.")
        return RagResult(retrieved_chunks=[], max_score=0.0, used_fallback=False)

    # 3. Determine the filter
    # If L1 strongly believes this is BOXING, we tell ChromaDB to only return BOXING chunks.
    # Otherwise, we search across all martial arts.
    sport_filter = None
    if intent_result.sport in (SportType.BOXING, SportType.KICKBOXING, SportType.KARATE):
        sport_filter = intent_result.sport.value.lower()

    # 4. Execute Search
    logger.info("Executing RAG search", extra={"query": search_query, "filter": sport_filter})
    
    try:
        raw_results = search(query=search_query, top_k=top_k, sport=sport_filter)
        
        if not raw_results:
            return RagResult(retrieved_chunks=[], max_score=0.0, used_fallback=False)
            
        # Extract the text chunks and calculate the max similarity score
        chunks = [res['text'] for res in raw_results]
        max_score = max(res['score'] for res in raw_results)
        
        # Important: MRR evaluation showed fine-tuned embeddings aren't perfect yet.
        # We will retrieve the chunks, but track the max_score. 
        # Level 5 (Gemini) can look at the score and decide if the context is actually relevant.
        
        return RagResult(
            retrieved_chunks=chunks,
            max_score=max_score,
            used_fallback=False
        )
        
    except Exception as exc:
        logger.error("RAG retrieval failed catastrophically", exc_info=exc)
        # We return gracefully so the Chatbot can still answer using Gemini's baseline knowledge
        return RagResult(retrieved_chunks=[], max_score=0.0, used_fallback=True)


if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    setup_logging()
    
    # Needs the ChromaDB instantiated in embeddings/vectorstore
    print("\n--- Testing Technique Search (Boxing) ---")
    intent1 = IntentResult(
        category=IntentCategory.TECHNIQUE,
        sport=SportType.BOXING,
        confidence=0.9,
        extracted_entities=[]
    )
    res1 = retrieve_context("how do I throw a jab", intent_result=intent1)
    print(f"Max Score: {res1.max_score}")
    for i, c in enumerate(res1.retrieved_chunks):
        print(f"Result {i+1}: {c[:100]}...")
        
    print("\n--- Testing Generic Chat (Should Skip) ---")
    intent2 = IntentResult(
        category=IntentCategory.GENERIC_CHAT,
        sport=SportType.UNKNOWN,
        confidence=1.0,
        extracted_entities=[]
    )
    res2 = retrieve_context("hi there", intent_result=intent2)
    print(res2)

    print("\n--- Testing Vision Augmentation (Karate) ---")
    intent3 = IntentResult(
        category=IntentCategory.TECHNIQUE,
        sport=SportType.KARATE,
        confidence=0.8,
        extracted_entities=[]
    )
    vision3 = VisionResult(
        description="Fighter performing a high roundhouse kick targeting the head.",
        extracted_techniques=["Mawashi Geri", "High Kick"],
        confidence=0.95
    )
    res3 = retrieve_context("is this legal?", intent_result=intent3, vision_result=vision3)
    print(f"Max Score: {res3.max_score}")
    for i, c in enumerate(res3.retrieved_chunks):
        safe_text = c[:100].encode('ascii', 'ignore').decode('ascii')
        print(f"Result {i+1}: {safe_text}...")
