"""
FightMind AI — Level 1: Intent Detection
========================================
Step 1.10 — Uses Gemini 1.5 Flash (via the google-genai SDK) with
Structured Outputs to classify incoming user queries.

This acts as a router for the rest of the 6-Level pipeline.
It tells upstream levels whether to search the vector database,
query the live events API, or just generate a generic response.
"""

import os
from enum import Enum
from typing import List

from google import genai
from pydantic import BaseModel, Field

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Structured Output Schemas (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class IntentCategory(str, Enum):
    TECHNIQUE = "TECHNIQUE"              # e.g., "how to throw a jab", "what is a liver hook"
    RULES_SCORING = "RULES_SCORING"      # e.g., "how many points for a knockdown", "is spinning backfist legal"
    FIGHTER_INFO = "FIGHTER_INFO"        # e.g., "who is mike tyson", "canelo alvarez record"
    LIVE_EVENT = "LIVE_EVENT"            # e.g., "when is the next ufc fight", "who did inoue fight last"
    GENERIC_CHAT = "GENERIC_CHAT"        # e.g., "hi", "who are you", "what can you do"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"      # e.g., "how to bake a cake", "what is photosynthesis"

class SportType(str, Enum):
    BOXING = "BOXING"
    KICKBOXING = "KICKBOXING"
    KARATE = "KARATE"
    GENERAL_MARTIAL_ARTS = "GENERAL_MARTIAL_ARTS"
    UNKNOWN = "UNKNOWN"

class IntentResult(BaseModel):
    category: IntentCategory = Field(
        description="The primary intent of the user's query."
    )
    sport: SportType = Field(
        description="The specific combat sport the query refers to. If unclear, use GENERAL_MARTIAL_ARTS or UNKNOWN."
    )
    confidence: float = Field(
        description="A confidence score between 0.0 and 1.0 representing how certain you are of the category."
    )
    extracted_entities: List[str] = Field(
        description="A list of specific entities found in the text (e.g., fighter names, event names, specific techniques). Provide empty list if none.",
        default_factory=list
    )


class _GeminiIntentResult(BaseModel):
    category: str = Field(
        description="The primary intent. Must be one of: TECHNIQUE, RULES_SCORING, FIGHTER_INFO, LIVE_EVENT, GENERIC_CHAT, OUT_OF_DOMAIN"
    )
    sport: str = Field(
        description="The specific combat sport. Must be one of: BOXING, KICKBOXING, KARATE, GENERAL_MARTIAL_ARTS, UNKNOWN"
    )
    confidence: float = Field(
        description="A confidence score between 0.0 and 1.0 representing how certain you are of the category."
    )
    extracted_entities: List[str] = Field(
        description="A list of specific entities found in the text (e.g., fighter names, event names). Provide empty list if none.",
        default_factory=list
    )

# ─────────────────────────────────────────────────────────────────────────────
# System Instruction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the central router (Level 1) for an AI chatbot dedicated strictly to Boxing, Kickboxing, and Karate.

Your job is to read the user's input and classify their intent into one of the provided categories.
You must absolutely respect the domain boundaries. If a user asks about something completely unrelated
to fighting, martial arts, or the chatbot itself, you MUST classify it as OUT_OF_DOMAIN.

Categories:
- TECHNIQUE: Questions about how to perform moves, stances, or physical training.
- RULES_SCORING: Questions about rulebooks, fouls, scoring systems, or legality.
- FIGHTER_INFO: Questions about specific fighters, their history, or styles.
- LIVE_EVENT: Questions about upcoming matches, recent results, or schedules.
- GENERIC_CHAT: Conversational filler like greetings ("hello", "thanks") or questions about your capabilities.
- OUT_OF_DOMAIN: Anything not related to combat sports or the chatbot.

Additionally:
- Try to determine the specific sport (BOXING, KICKBOXING, KARATE).
- Extract important entities (like "Canelo Alvarez", "UFC 300", "Liver Hook") as a list of strings.
- Provide a confidence score between 0.0 and 1.0.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Main Implementation
# ─────────────────────────────────────────────────────────────────────────────

def detect_intent(
    text: str,
    image_url: str | None = None,
    history: list | None = None
) -> IntentResult:
    """
    Analyzes the user's text using Gemini 1.5 Flash to determine the intent category.

    Args:
        text: The incoming text query from the user.
        image_url: Optional image URL (not used in text classification, but reserved for signature).
        history: Optional chat history (reserved for multi-turn context).

    Returns:
        A validated IntentResult Pydantic model with category, sport, entities, and confidence.
    """
    if not text or not text.strip():
        logger.warning("Empty text passed to detect_intent, defaulting to GENERIC_CHAT")
        return IntentResult(
            category=IntentCategory.GENERIC_CHAT,
            sport=SportType.UNKNOWN,
            confidence=1.0,
            extracted_entities=[]
        )

    # Requires GEMINI_API_KEY environment variable to be set
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    try:
        response = client.models.generate_content(
            model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
            contents=text,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_GeminiIntentResult,
                temperature=0.0  # Zero temperature for highly deterministic classification
            ),
        )
        
        parsed: _GeminiIntentResult = response.parsed
        
        # Safely convert strings to Enums
        try:
            cat_enum = IntentCategory(parsed.category)
        except ValueError:
            cat_enum = IntentCategory.GENERIC_CHAT
            
        try:
            sport_enum = SportType(parsed.sport)
        except ValueError:
            sport_enum = SportType.UNKNOWN
            
        result = IntentResult(
            category=cat_enum,
            sport=sport_enum,
            confidence=parsed.confidence,
            extracted_entities=parsed.extracted_entities
        )
        
        logger.info(
            "Intent detected",
            extra={
                "category": result.category.value,
                "sport": result.sport.value,
                "confidence": result.confidence,
                "entities": result.extracted_entities
            }
        )
        return result

    except Exception as exc:
        logger.error("Failed to detect intent with Gemini", exc_info=exc, extra={"text": text[:50]})
        
        # Safe fallback so pipeline doesn't crash on API failure
        return IntentResult(
            category=IntentCategory.GENERIC_CHAT,
            sport=SportType.UNKNOWN,
            confidence=0.0,
            extracted_entities=[]
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.core.logging_config import setup_logging
    
    load_dotenv()
    setup_logging()
    
    # Needs GEMINI_API_KEY in environment
    print("\n--- Testing Technique ---")
    res1 = detect_intent("how do I throw a liver hook to the body?")
    print(res1)
    
    import time
    time.sleep(2)
    
    print("\n--- Testing Live Event ---")
    res2 = detect_intent("when is naoya inoue fighting next?")
    print(res2)
    
    time.sleep(2)
    
    print("\n--- Testing Out of Domain ---")
    res3 = detect_intent("how long does it take to bake chocolate chip cookies?")
    print(res3)
