"""
FightMind AI — Level 6: Validation & Personalization
======================================================
Step 1.15 — The final guardrail before the answer reaches the user.

Takes the completed answer from L5 and reviews it using Gemini 2.0 Flash to detect 
hallucinations, ensure safety, and extract the user's estimated skill level for future RAG profile tracking.
"""

from enum import Enum
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from google import genai
from google.genai import types

from src.core.logging_config import get_logger
from src.pipeline.level3_rag import RagResult

logger = get_logger(__name__)
load_dotenv()

_client = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Structured Output Schema
# ─────────────────────────────────────────────────────────────────────────────

class SkillLevel(str, Enum):
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"
    UNKNOWN = "UNKNOWN"

# We can run into the Enum validation bug with pydantic/genai here too.
# So we use string fields for the LLM Output, and map it later.
class _GeminiValidationResult(BaseModel):
    is_safe: bool = Field(
        description="True if the response is safe. False if it encourages street fighting, illegal acts, or dangerous, untrained techniques."
    )
    hallucination_detected: bool = Field(
        description="True if the LLM generated answer claims false facts or contradicts the provided RAG text chunks."
    )
    detected_skill_level: str = Field(
        description="Must be exactly one of: BEGINNER, INTERMEDIATE, ADVANCED, or UNKNOWN. Estimate based on the complexity of the user's query."
    )


class ValidationResult(BaseModel):
    is_safe: bool
    hallucination_detected: bool
    detected_skill_level: SkillLevel


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Engineering
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are FightMind AI's internal safety and quality auditor.
Review the following User Query, the verified Knowledge Base Text, and the final Generated Answer.

Your job is to:
1. Detect Hallucinations: If the Generated Answer states facts that contradict the Knowledge Base Text, flag hallucination_detected as True. (If knowledge base is empty, just check for obvious martial arts falsehoods).
2. Assess Safety: If the Generated Answer encourages street fighting, illegal acts, or dangerous techniques without proper warnings, flag is_safe as False.
3. Profile Skill: Estimate the user's combat sports experience based on the phrasing and technical depth of their query (BEGINNER, INTERMEDIATE, ADVANCED, or UNKNOWN).

Return strictly JSON according to the schema.
"""

def _build_validation_prompt(query: str, rag_res: RagResult, generated_answer: str) -> str:
    """Combines the query, context, and generated answer for review."""
    
    prompt_parts = [
        f"USER QUERY:\n{query}\n"
    ]
    
    if rag_res and rag_res.retrieved_chunks:
        chunks_str = "\n".join([f"- {c}" for c in rag_res.retrieved_chunks])
        prompt_parts.append(f"\nVERIFIED KNOWLEDGE BASE TEXT:\n{chunks_str}\n")
    else:
        prompt_parts.append("\nVERIFIED KNOWLEDGE BASE TEXT:\nNone provided.\n")
        
    prompt_parts.append(f"\nGENERATED ANSWER TO REVIEW:\n{generated_answer}\n")
    
    return "\n".join(prompt_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Implementation
# ─────────────────────────────────────────────────────────────────────────────

def validate_answer(query: str, rag_res: RagResult, generated_answer: str) -> ValidationResult:
    """
    Final pipeline step. Reviews the L5 answer for safety and extracts user profile data.
    """
    logger.info("Level 6: Validating answer and extracting user skill profile...")
    
    compiled_prompt = _build_validation_prompt(query, rag_res, generated_answer)
    
    try:
        response = get_client().models.generate_content(
            model='gemini-2.0-flash',
            contents=compiled_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1, # strict analysis
                response_mime_type="application/json",
                response_schema=_GeminiValidationResult,
            ),
        )
        
        raw_res = response.parsed
        if raw_res is None:
            raise ValueError("Gemini SDK returned None during validation Pydantic parsing.")
            
        # Map string enum safely
        skill_str = raw_res.detected_skill_level.upper()
        skill_enum = SkillLevel.UNKNOWN
        if skill_str in [e.value for e in SkillLevel]:
            skill_enum = SkillLevel(skill_str)
            
        result = ValidationResult(
            is_safe=raw_res.is_safe,
            hallucination_detected=raw_res.hallucination_detected,
            detected_skill_level=skill_enum
        )
        
        logger.info(
            "Validation complete", 
            extra={
                "safe": result.is_safe, 
                "hallucination": result.hallucination_detected, 
                "skill": result.detected_skill_level.value
            }
        )
        
        return result

    except Exception as exc:
        logger.error("Level 6 validation failed, defaulting to safe/unknown", exc_info=exc)
        # Fail open rather than crashing the whole bot
        return ValidationResult(
            is_safe=True,
            hallucination_detected=False,
            detected_skill_level=SkillLevel.UNKNOWN
        )


if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    setup_logging()
    
    print("\n--- Testing Level 6 Validation (Safe) ---")
    safe_q = "How do I throw a jab?"
    safe_rag = RagResult(retrieved_chunks=["Keep your chin tucked."], max_score=0.9, used_fallback=False)
    safe_ans = "To throw a jab, extend your lead hand straight out while keeping your chin tucked for protection."
    res1 = validate_answer(safe_q, safe_rag, safe_ans)
    print(f"Safe: {res1.is_safe} | Hallucinates: {res1.hallucination_detected} | Skill: {res1.detected_skill_level.value}")
    
    print("\n--- Testing Level 6 Validation (Hallucination) ---")
    bad_q = "Who won UFC 300 main event?"
    bad_rag = RagResult(retrieved_chunks=["UFC 300 main event was Alex Pereira vs Jamahal Hill."], max_score=0.9, used_fallback=False)
    bad_ans = "Conor McGregor won UFC 300 by knockout."
    res2 = validate_answer(bad_q, bad_rag, bad_ans)
    print(f"Safe: {res2.is_safe} | Hallucinates: {res2.hallucination_detected} | Skill: {res2.detected_skill_level.value}")
