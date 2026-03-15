"""
FightMind AI — Level 4: Live Events
===================================
Step 1.13 — Connects the Level 1 Intent to the live Sports API.

If the user is asking about a live event (e.g., "when is UFC 300?"),
this module fetches the schedule from TheSportsDB and formats it into
a text block for the LLM to read and answer from.
"""
from pydantic import BaseModel, Field

from src.core.logging_config import get_logger
from src.pipeline.level1_intent import IntentCategory, IntentResult
from src.data_collection.sports_api import fetch_events

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Structured Output Schema
# ─────────────────────────────────────────────────────────────────────────────

class EventsResult(BaseModel):
    event_context: str = Field(
        description="A formatted text block containing full schedule details from the live API."
    )
    has_events: bool = Field(
        description="True if the API found matching events and populated the context."
    )
    used_fallback: bool = Field(
        description="True if the API failed or timed out, forcing reliance on internal LLM knowledge."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Implementation
# ─────────────────────────────────────────────────────────────────────────────

def _format_event_list(events: list[dict], search_entities: list[str]) -> str:
    """
    Helps format raw JSON events from TheSportsDB into a readable string block for Gemini.
    Filters the events to only include those matching the extracted entities (if any).
    """
    if not events:
        return ""
        
    lines = ["--- LIVE EVENT DATA ---"]
    match_count = 0
    
    # Simple lower-case matching
    entities_lower = [e.lower() for e in search_entities]
    
    for e in events:
        name = e.get("title", "Unknown Event")
        promotion = e.get("promotion", "") or ""
        fighters = e.get("fighters", "") or ""
        
        # If entities array has specific fighters/promotions, filter.
        # Otherwise, if empty array [] or ["UFC"], we just show everything.
        is_match = True
        
        # We only strictly filter if they asked for a specific person or weird promotion
        if entities_lower and "ufc" not in entities_lower and "ultimate fighting championship" not in entities_lower:
            text_to_search = f"{name} {promotion} {fighters}".lower()
            is_match = any(ent in text_to_search for ent in entities_lower)
            
        if not is_match:
            continue
            
        date = e.get("date", "Unknown Date")
        time = e.get("time", "Unknown Time")
        status = e.get("status", "Unknown Status")
        venue = e.get("venue", "Unknown Venue")
        
        lines.append(f"Event: {name} | Date: {date} at {time} | Status: {status} | Venue: {venue}")
        match_count += 1
        
        # Hard limit to 15 events to prevent LLM context overflow
        if match_count >= 15:
            break
            
    if match_count == 0:
        return ""
        
    lines.append("-----------------------")
    return "\n".join(lines)


def fetch_live_context(intent_result: IntentResult) -> EventsResult:
    """
    Fetches live sports data IF AND ONLY IF Level 1 flagged it as a LIVE_EVENT.
    Uses the 3-source fallback orchestrator built in Phase 1B.
    """
    # 1. Short Circuit Check
    if intent_result.category != IntentCategory.LIVE_EVENT:
        logger.debug("Skipping Live Events API", extra={"reason": "Not a LIVE_EVENT intent"})
        return EventsResult(event_context="", has_events=False, used_fallback=False)
        
    logger.info("Live Event intent detected, querying external APIs...")
    
    try:
        # fetch_events pulls from TheSportsDB -> API-Sports -> Web Scrape automatically
        events_found = fetch_events()
                    
        # 3. Format result
        if not events_found:
            logger.info("No matching events found in live API")
            return EventsResult(
                event_context="No live event schedule found for this query.",
                has_events=False,
                used_fallback=False
            )
            
        # Filter the universal list down to just what the user asked about
        context_str = _format_event_list(events_found, intent_result.extracted_entities)
        
        if not context_str:
            return EventsResult(
                event_context=f"No live events found matching entities: {intent_result.extracted_entities}",
                has_events=False,
                used_fallback=False
            )
            
        return EventsResult(
            event_context=context_str,
            has_events=True,
            used_fallback=False
        )

    except Exception as exc:
        logger.error("Live Events API failed", exc_info=exc)
        return EventsResult(
            event_context="Live sports API is currently unavailable.",
            has_events=False,
            used_fallback=True
        )


if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    from dotenv import load_dotenv
    
    load_dotenv()
    setup_logging()
    
    print("\n--- Testing UFC Query ---")
    intent1 = IntentResult(
        category=IntentCategory.LIVE_EVENT,
        sport="UNKNOWN",
        confidence=0.9,
        extracted_entities=["UFC"]
    )
    res1 = fetch_live_context(intent1)
    print(res1.event_context)
    
    print("\n--- Testing Specific Entity (Non-UFC) ---")
    intent2 = IntentResult(
        category=IntentCategory.LIVE_EVENT,
        sport="BOXING",
        confidence=0.9,
        extracted_entities=["Canelo"]
    )
    res2 = fetch_live_context(intent2)
    print(f"Has Events: {res2.has_events}")
    print(res2.event_context)
    
    print("\n--- Testing Non-Event Intent (Skipped) ---")
    intent3 = IntentResult(
        category=IntentCategory.TECHNIQUE,
        sport="BOXING",
        confidence=0.9,
        extracted_entities=[]
    )
    res3 = fetch_live_context(intent3)
    print(f"Context: '{res3.event_context}'")
