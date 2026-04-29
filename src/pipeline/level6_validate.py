"""
FightMind AI — Level 6: Validation & Personalization
======================================================
Step 4.5 — Replaced Gemini API call with a fully LOCAL rule-based checker.

This eliminates 1 Gemini API call per request (saves ~33% API cost).

Three local checks are performed:
1. Safety Check     — keyword scan for dangerous/illegal content in the answer
2. Hallucination    — word-overlap score between answer and RAG source chunks
3. Skill Profiling  — keyword scan of the user query to estimate experience level
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel

from src.core.logging_config import get_logger
from src.pipeline.level3_rag import RagResult

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output Schema (unchanged — same contract as before)
# ─────────────────────────────────────────────────────────────────────────────

class SkillLevel(str, Enum):
    BEGINNER     = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED     = "ADVANCED"
    UNKNOWN      = "UNKNOWN"


class ValidationResult(BaseModel):
    is_safe: bool
    hallucination_detected: bool
    detected_skill_level: SkillLevel


# ─────────────────────────────────────────────────────────────────────────────
# Rule Sets
# ─────────────────────────────────────────────────────────────────────────────

# Words that indicate the answer may be promoting dangerous/illegal behaviour
_UNSAFE_KEYWORDS = [
    "street fight", "kill", "murder", "illegal weapon", "eye gouge",
    "groin strike", "fish hook", "bite", "headbutt", "use a knife",
    "attack someone", "how to hurt", "how to assault", "unregulated fight",
    "no rules", "backyard fight", "unsanctioned",
]

# Query keywords that indicate an advanced user
_ADVANCED_KEYWORDS = [
    "southpaw", "orthodox", "counterpunch", "combination", "footwork drill",
    "head movement", "parry", "shoulder roll", "philly shell", "peek-a-boo",
    "clinch work", "thai plum", "teep", "switch kick", "spinning back",
    "compound combination", "level change", "cage control", "ground and pound",
    "guard pass", "sprawl", "underhook", "overhook", "double leg",
]

# Query keywords that indicate a beginner user
_BEGINNER_KEYWORDS = [
    "what is", "how do i", "explain", "beginner", "newbie", "start",
    "first time", "never trained", "basics", "basic", "introduction",
    "simple", "easy", "learn", "how to", "i am new",
]


# ─────────────────────────────────────────────────────────────────────────────
# Local Checkers
# ─────────────────────────────────────────────────────────────────────────────

def _check_safety(answer: str) -> bool:
    """
    Returns True (safe) if no unsafe keywords are found in the answer.
    O(n) keyword scan — runs in microseconds.
    """
    answer_lower = answer.lower()
    for keyword in _UNSAFE_KEYWORDS:
        if keyword in answer_lower:
            logger.warning("Level 6: Unsafe keyword detected in answer", extra={"keyword": keyword})
            return False
    return True


def _check_hallucination(answer: str, rag_res: Optional[RagResult]) -> bool:
    """
    Estimates hallucination using a word-overlap score.

    Logic:
    - If no RAG chunks were used, we can't check → return False (benefit of doubt)
    - Tokenise the answer and each RAG chunk into word sets
    - If the answer shares < 5% of its unique words with ALL chunks combined,
      it's likely the LLM ignored the context → flag as hallucination

    This is a simple but effective heuristic. False positive rate is low
    because any answer that uses even a few domain-specific terms from the
    RAG chunks will pass the threshold.
    """
    if not rag_res or not rag_res.retrieved_chunks:
        return False  # No RAG context to compare against → cannot detect

    # Build a set of all meaningful words from RAG chunks (ignore short stop words)
    rag_words: set[str] = set()
    for chunk in rag_res.retrieved_chunks:
        for word in chunk.lower().split():
            clean = word.strip(".,!?\"'();:")
            if len(clean) > 3:  # ignore 'the', 'is', 'a', etc.
                rag_words.add(clean)

    if not rag_words:
        return False

    # Build word set from the answer
    answer_words: set[str] = set()
    for word in answer.lower().split():
        clean = word.strip(".,!?\"'();:")
        if len(clean) > 3:
            answer_words.add(clean)

    if not answer_words:
        return False

    # Overlap ratio: what fraction of answer words appear in the RAG corpus
    overlap = len(answer_words & rag_words)
    overlap_ratio = overlap / len(answer_words)

    hallucination = overlap_ratio < 0.05  # less than 5% overlap → suspicious
    if hallucination:
        logger.warning(
            "Level 6: Low RAG overlap — possible hallucination",
            extra={"overlap_ratio": round(overlap_ratio, 3)}
        )
    return hallucination


def _detect_skill_level(query: str) -> SkillLevel:
    """
    Estimates user skill level from the vocabulary used in their query.
    Simple keyword scan — runs in microseconds.
    """
    query_lower = query.lower()

    # Check advanced first (takes priority)
    for keyword in _ADVANCED_KEYWORDS:
        if keyword in query_lower:
            return SkillLevel.ADVANCED

    for keyword in _BEGINNER_KEYWORDS:
        if keyword in query_lower:
            return SkillLevel.BEGINNER

    return SkillLevel.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point (same signature as before — drop-in replacement)
# ─────────────────────────────────────────────────────────────────────────────

def validate_answer(query: str, rag_res: Optional[RagResult], generated_answer: str) -> ValidationResult:
    """
    Final pipeline guardrail. Runs fully locally — zero API calls.

    Args:
        query:            The original user query.
        rag_res:          RAG retrieval result (used for hallucination check).
        generated_answer: The answer produced by Level 5.

    Returns:
        ValidationResult with is_safe, hallucination_detected, detected_skill_level.
    """
    logger.info("Level 6: Running local validation checks...")

    is_safe            = _check_safety(generated_answer)
    hallucination      = _check_hallucination(generated_answer, rag_res)
    skill              = _detect_skill_level(query)

    result = ValidationResult(
        is_safe=is_safe,
        hallucination_detected=hallucination,
        detected_skill_level=skill,
    )

    logger.info(
        "Level 6 local validation complete",
        extra={
            "safe": result.is_safe,
            "hallucination": result.hallucination_detected,
            "skill": result.detected_skill_level.value,
        }
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Manual Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    setup_logging()

    print("\n--- Test 1: Safe answer (beginner query) ---")
    res1 = validate_answer(
        query="How do I throw a jab?",
        rag_res=RagResult(retrieved_chunks=["Keep your chin tucked. Extend your lead hand straight out."], max_score=0.9, used_fallback=False),
        generated_answer="To throw a jab, extend your lead hand straight out while keeping your chin tucked for protection."
    )
    print(f"Safe: {res1.is_safe} | Hallucinates: {res1.hallucination_detected} | Skill: {res1.detected_skill_level.value}")

    print("\n--- Test 2: Unsafe answer ---")
    res2 = validate_answer(
        query="How do I win a street fight?",
        rag_res=None,
        generated_answer="In a street fight with no rules, you can eye gouge or bite your opponent to escape."
    )
    print(f"Safe: {res2.is_safe} | Hallucinates: {res2.hallucination_detected} | Skill: {res2.detected_skill_level.value}")

    print("\n--- Test 3: Hallucination detection ---")
    res3 = validate_answer(
        query="Who won UFC 300 main event?",
        rag_res=RagResult(retrieved_chunks=["UFC 300 main event was Alex Pereira vs Jamahal Hill."], max_score=0.9, used_fallback=False),
        generated_answer="Conor McGregor won UFC 300 by knockout in round one against Khabib Nurmagomedov."
    )
    print(f"Safe: {res3.is_safe} | Hallucinates: {res3.hallucination_detected} | Skill: {res3.detected_skill_level.value}")

    print("\n--- Test 4: Advanced user query ---")
    res4 = validate_answer(
        query="How do I use the philly shell to set up a counter left hook?",
        rag_res=None,
        generated_answer="Drop your lead shoulder, roll under the jab, and fire the left hook off the back foot."
    )
    print(f"Safe: {res4.is_safe} | Hallucinates: {res4.hallucination_detected} | Skill: {res4.detected_skill_level.value}")
