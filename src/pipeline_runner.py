"""
FightMind AI — Core Pipeline Runner
===================================
Step 1.16 — The central orchestrator.

This module exposes the `run_pipeline()` function, which chains the 6 pipeline 
levels together and returns a unified JSON output intended for the FastAPI backend.
"""

from typing import Optional
from pydantic import BaseModel, Field

from src.core.logging_config import get_logger
from src.pipeline.level1_intent import IntentCategory, IntentResult
from src.pipeline.level6_validate import SkillLevel

# Import the 6 pipeline stages
from src.pipeline.level1_intent import detect_intent
from src.pipeline.level2_vision import analyze_image
from src.pipeline.level3_rag import retrieve_context
from src.pipeline.level4_events import fetch_live_context
from src.pipeline.level5_llm import generate_answer
from src.pipeline.level6_validate import validate_answer

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output Schema (API Contract)
# ─────────────────────────────────────────────────────────────────────────────

class ChatbotResponse(BaseModel):
    """The final API contract that is sent from the Python model back to the Java/React layers."""
    query: str
    answer: str
    confidence_score: float
    
    # Metadata for UI/UX
    detected_intent: str
    detected_sport: str
    user_skill_level: str
    
    # Debugging / System flags
    used_vision: bool
    used_rag: bool
    used_live_events: bool
    hallucination_flag: bool
    fallback_engaged: bool


# ─────────────────────────────────────────────────────────────────────────────
# Implementation
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(query: str, image_url: Optional[str] = None) -> ChatbotResponse:
    """
    Executes the 6-Level AI Pipeline sequentially.
    
    1. Intent Classification
    2. Vision Processing (if image_url is provided)
    3. RAG Retrieval
    4. Live Events API
    5. LLM Answer Generation
    6. Answer Validation & Personalization
    """
    logger.info("Starting pipeline execution", extra={"query_length": len(query), "has_image": bool(image_url)})
    
    # ---------------------------------------------------------
    # LEVEL 1: Intent
    # ---------------------------------------------------------
    intent_res = detect_intent(query)
    logger.debug("Level 1 complete", extra={"intent": intent_res.category.value})
    
    # Short-Circuit Mechanism
    if intent_res.category == IntentCategory.OUT_OF_DOMAIN:
        logger.warning("Query out of domain, terminating pipeline early.")
        return ChatbotResponse(
            query=query,
            answer="I am FightMind AI, a specialized combat sports coach. I can only answer questions related to martial arts, technique, training, and combat sports events. Please ask me a fight-related question!",
            confidence_score=1.0,
            detected_intent=intent_res.category.value,
            detected_sport=intent_res.sport.value,
            user_skill_level=SkillLevel.UNKNOWN.value,
            used_vision=False,
            used_rag=False,
            used_live_events=False,
            hallucination_flag=False,
            fallback_engaged=False
        )
        
    # ---------------------------------------------------------
    # LEVEL 2: Vision
    # ---------------------------------------------------------
    vision_res = None
    if image_url:
        vision_res = analyze_image(image_url, query)
        logger.debug("Level 2 Vision complete", extra={"success": vision_res is not None})
        
    # ---------------------------------------------------------
    # LEVEL 3: RAG
    # ---------------------------------------------------------
    # We always query RAG unless the user asked something trivial like "Hello"
    if intent_res.category == IntentCategory.GENERIC_CHAT:
        rag_res = None
        logger.debug("Skipping Level 3 RAG (Generic Chat)")
    else:
        rag_res = retrieve_context(query, intent_res, vision_res)
        logger.debug("Level 3 RAG complete", extra={"chunks_retrieved": len(rag_res.retrieved_chunks) if rag_res else 0})
        
    # ---------------------------------------------------------
    # LEVEL 4: Live Events
    # ---------------------------------------------------------
    events_res = fetch_live_context(intent_res)
    logger.debug("Level 4 Events complete", extra={"events_found": events_res.has_events})
    
    # ---------------------------------------------------------
    # LEVEL 5: LLM Synthesis
    # ---------------------------------------------------------
    llm_res = generate_answer(query, intent_res, vision_res, rag_res, events_res)
    logger.debug("Level 5 LLM Synthesis complete", extra={"confidence": llm_res.confidence})
    
    # ---------------------------------------------------------
    # LEVEL 6: Validation
    # ---------------------------------------------------------
    if llm_res.used_fallback:
        # If generation failed, don't validate the fallback string
        validation_res = None
        skill_str = SkillLevel.UNKNOWN.value
        final_answer = llm_res.answer
        hallucination = False
        logger.debug("Skipping Level 6 Validation due to L5 Fallback")
    else:
        validation_res = validate_answer(query, rag_res, llm_res.answer)
        skill_str = validation_res.detected_skill_level.value
        hallucination = validation_res.hallucination_detected
        
        final_answer = llm_res.answer
        if hallucination:
            logger.warning("Level 6 detected hallucination in final answer.")
            final_answer += "\n\n*(Disclaimer: Portions of this answer may refer to information outside of our verified training database.)*"
            
        if not validation_res.is_safe:
            logger.warning("Level 6 flagged answer as unsafe.")
            final_answer = "This query touches upon dangerous, illegal, or unsanctioned combat topics. FightMind AI is dedicated to safe and legal martial arts practice. I cannot provide this information."
            hallucination = False
    
    # ---------------------------------------------------------
    # Final Construction
    # ---------------------------------------------------------
    response = ChatbotResponse(
        query=query,
        answer=final_answer,
        confidence_score=llm_res.confidence,
        detected_intent=intent_res.category.value,
        detected_sport=intent_res.sport.value,
        user_skill_level=skill_str,
        used_vision=vision_res is not None,
        used_rag=rag_res is not None and len(rag_res.retrieved_chunks) > 0,
        used_live_events=events_res.has_events,
        hallucination_flag=hallucination,
        fallback_engaged=llm_res.used_fallback
    )
    
    logger.info("Pipeline executed successfully", extra={"intent": intent_res.category.value, "skill": skill_str})
    return response


if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    from dotenv import load_dotenv
    import json
    
    load_dotenv()
    setup_logging()
    
    print("\n==================================")
    print("  FIGHTMIND AI - PIPELINE TEST")
    print("==================================\n")
    
    test_query = "What is a 1-2 combo in boxing? Explain it to me like I just walked into a gym."
    print(f"User Query: '{test_query}'")
    print("Running through Levels 1 - 6...")
    
    final_output = run_pipeline(test_query)
    
    print("\n--- JSON OUTPUT RESPONSE ---\n")
    print(final_output.model_dump_json(indent=2))
