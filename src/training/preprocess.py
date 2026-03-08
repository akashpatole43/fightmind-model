"""
FightMind AI — Raw Data Preprocessor
======================================
Step 1.6 — Loads scraped JSON files from data/raw/, cleans and chunks
the text into overlapping windows, tags each chunk with metadata, and
saves the result to data/processed/chunks.json.

Output format (one record per chunk):
{
    "chunk_id":    "wikipedia_0042",   # unique ID: <source>_<index>
    "text":        "A jab is a quick...",
    "sport":       "boxing",           # detected from content
    "source":      "wikipedia",        # original data source
    "doc_title":   "Jab (boxing)",     # parent article/page title
    "doc_url":     "https://...",      # original URL
    "chunk_index": 42,                 # position within the document
    "total_chunks": 3,                 # total chunks from this document
}

Chunking Strategy:
  - Chunk size  : CHUNK_SIZE tokens (approx. chars / 4)
  - Overlap     : CHUNK_OVERLAP chars from end of previous chunk
  - Why overlap : prevents a concept being split across two chunks with
                  no shared context, which hurts RAG retrieval accuracy.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# Path breakdown from src/training/preprocess.py:
#   parents[0] = src/training
#   parents[1] = src
#   parents[2] = fightmind-model  ← project root ✓
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RAW_DIR        = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR  = _PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chunking Parameters
# ─────────────────────────────────────────────────────────────────────────────

# ~400 tokens × 4 chars/token ≈ 1600 chars per chunk
# This fits well within ChromaDB and sentence-transformer context windows
CHUNK_SIZE    = 1_600   # characters
CHUNK_OVERLAP = 200     # character overlap between consecutive chunks
MIN_CHUNK_LEN = 80      # discard chunks shorter than this (header noise)

# ─────────────────────────────────────────────────────────────────────────────
# Sport Detection
# ─────────────────────────────────────────────────────────────────────────────

# Keyword sets for auto-detecting which sport a document belongs to.
# Order matters: if a document matches multiple, the first match wins.
_SPORT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("karate", [
        "karate", "kata", "kumite", "kihon", "dojo", "sensei",
        "shotokan", "goju", "wado", "kyokushin", "zenkutsu",
        "mae-geri", "mawashi", "oi-zuki", "gyaku-zuki", "kyu",
        "dan rank", "belt (martial arts)", "gi (uniform)",
    ]),
    ("kickboxing", [
        "kickboxing", "roundhouse kick", "low kick", "muay thai",
        "k-1", "glory kickboxing", "savate", "dutch kickboxing",
        "american kickboxing", "point fighting", "teep", "knee strike",
        "elbow strike",
    ]),
    ("boxing", [
        "boxing", "jab", "uppercut", "hook", "cross", "clinch",
        "knockout", "wbc", "wba", "ibf", "wbo", "southpaw",
        "heavyweight", "flyweight", "welterweight", "featherweight",
        "prizefighting", "purse", "ring",
    ]),
]

_GENERAL_SPORTS = {"combat sport", "martial arts", "self-defense", "conditioning"}


def _detect_sport(title: str, text: str) -> str:
    """
    Detect which sport a document primarily covers.

    Strategy:
        1. Check `title` for exact keyword matches (faster, more reliable)
        2. Fall back to counting keyword occurrences in first 600 chars of `text`
        3. Default to "general" if no sport detected

    Args:
        title: Document title (e.g. "Jab (boxing)")
        text:  Document body text

    Returns:
        "boxing" | "kickboxing" | "karate" | "general"
    """
    combined = (title + " " + text[:600]).lower()

    for sport, keywords in _SPORT_KEYWORDS:
        if any(kw in combined for kw in keywords):
            return sport

    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

# Compile regex patterns once at module level for performance
_MULTI_NEWLINES  = re.compile(r"\n{3,}")
_MULTI_SPACES    = re.compile(r"[ \t]{2,}")
_CITATION_NUMS   = re.compile(r"\[\d+\]")           # Wikipedia [1], [2], etc.
_SECTION_EQUALS  = re.compile(r"={2,}[^=]+={2,}")   # == Section Title ==  (2+ = on each side)
_EDIT_LINKS      = re.compile(r"\[edit\]", re.IGNORECASE)
_BARE_URLS       = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """
    Normalize raw scraped text for chunking and embedding.

    Operations applied (in order):
        1. Unicode normalization (NFC) — fixes encoding artefacts
        2. Remove Wikipedia citation markers like [1], [23]
        3. Remove Wikipedia section markers == Title ==
        4. Remove [edit] links from Wikipedia
        5. Collapse 3+ consecutive newlines → 2 (preserve paragraph breaks)
        6. Collapse multiple spaces/tabs → single space
        7. Strip leading/trailing whitespace

    Args:
        text: Raw scraped text

    Returns:
        Cleaned text string. Empty string if input is empty/None.
    """
    if not text:
        return ""

    # Step 1: Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2–4: Remove Wikipedia-specific noise
    text = _CITATION_NUMS.sub("", text)
    text = _SECTION_EQUALS.sub("", text)
    text = _EDIT_LINKS.sub("", text)
    text = _BARE_URLS.sub("", text)

    # Step 5–7: Whitespace normalization
    text = _MULTI_NEWLINES.sub("\n\n", text)
    text = _MULTI_SPACES.sub(" ", text)
    text = text.strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping fixed-size character windows.

    Boundary-aware splitting:
        - Tries to split on the nearest paragraph break (\\n\\n) within ±10%
          of chunk_size to avoid cutting mid-sentence
        - Falls back to sentence boundary (". ") if no paragraph break found
        - Hard-splits at chunk_size as last resort

    Args:
        text:       Cleaned text to split
        chunk_size: Target chunk size in characters
        overlap:    Characters to carry over from the previous chunk

    Returns:
        List of non-empty string chunks, each ≥ MIN_CHUNK_LEN characters.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take everything remaining
            chunk = text[start:].strip()
            if len(chunk) >= MIN_CHUNK_LEN:
                chunks.append(chunk)
            break

        # Look for a paragraph break within the last 10% of the window
        search_from = end - chunk_size // 10
        para_pos = text.rfind("\n\n", search_from, end)
        if para_pos != -1:
            end = para_pos
        else:
            # Fall back to sentence boundary
            sent_pos = text.rfind(". ", search_from, end)
            if sent_pos != -1:
                end = sent_pos + 1   # include the period

        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)

        # Move start forward, retaining overlap characters for context continuity
        start = max(start + 1, end - overlap)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Per-Source Processors
# ─────────────────────────────────────────────────────────────────────────────

def process_records(records: list[dict], default_source: str) -> list[dict]:
    """
    Clean, chunk, and enrich a list of raw scraped records.

    Each input record is expected to have:
        { "title": str, "text": str, "url": str, "source": str }

    Each output chunk has:
        { chunk_id, text, sport, source, doc_title, doc_url,
          chunk_index, total_chunks }

    Skips records with:
        - Missing or empty text (logged at WARNING)
        - Text shorter than MIN_CHUNK_LEN after cleaning (too short to embed)

    Args:
        records:        List of raw dicts from a JSON file
        default_source: Fallback source label if record has no "source" key

    Returns:
        List of chunk dicts ready to write to chunks.json
    """
    all_chunks: list[dict] = []
    skipped = 0

    for record in records:
        title  = record.get("title", "Unknown")
        text   = record.get("text", "")
        url    = record.get("url", "")
        source = record.get("source", default_source)

        if not text or not text.strip():
            logger.warning("Skipping record with empty text", extra={"title": title, "source": source})
            skipped += 1
            continue

        cleaned = clean_text(text)
        if len(cleaned) < MIN_CHUNK_LEN:
            logger.warning(
                "Skipping record — text too short after cleaning",
                extra={"title": title, "chars": len(cleaned), "min": MIN_CHUNK_LEN},
            )
            skipped += 1
            continue

        sport  = _detect_sport(title, cleaned)
        chunks = chunk_text(cleaned)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{source}_{len(all_chunks):05d}"
            all_chunks.append({
                "chunk_id":    chunk_id,
                "text":        chunk,
                "sport":       sport,
                "source":      source,
                "doc_title":   title,
                "doc_url":     url,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            })

        logger.debug(
            "Document chunked",
            extra={"title": title, "sport": sport, "chunks": len(chunks), "chars": len(cleaned)},
        )

    logger.debug(
        "Source processing complete",
        extra={"source": default_source, "records": len(records), "skipped": skipped, "chunks": len(all_chunks)},
    )
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# File Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(filename: str) -> list[dict]:
    """
    Load a JSON file from data/raw/.

    Returns empty list if file does not exist (logged at WARNING)
    or if the file is malformed (logged at ERROR).
    """
    path = RAW_DIR / filename
    if not path.exists():
        logger.warning("Raw data file not found — run scraper first", extra={"file": filename, "path": str(path)})
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("Expected a JSON array in file", extra={"file": filename, "type": type(data).__name__})
            return []

        logger.info("Raw file loaded", extra={"file": filename, "records": len(data)})
        return data

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON file", exc_info=exc, extra={"file": filename})
        return []
    except (IOError, PermissionError) as exc:
        logger.error("Failed to read file", exc_info=exc, extra={"file": filename})
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Save Output
# ─────────────────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], filename: str = "chunks.json") -> Path:
    """
    Save processed chunks to data/processed/<filename>.

    Raises:
        IOError / PermissionError: logged at CRITICAL and re-raised.
    """
    path = PROCESSED_DIR / filename
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(
            "Chunks saved",
            extra={"file_name": filename, "chunk_count": len(chunks), "path": str(path)},
        )
    except (IOError, PermissionError) as exc:
        logger.critical(
            "Failed to write chunks file — data will be lost",
            exc_info=exc,
            extra={"file_name": filename, "path": str(path)},
        )
        raise
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing() -> dict[str, int]:
    """
    Load all raw JSON files, process them into chunks, and save to
    data/processed/chunks.json.

    Processing order:
        1. wikipedia.json         — largest source, most context
        2. official_rules.json    — dense rule text
        3. youtube_transcripts.json — conversational tutorial content
        4. live_events.json       — kept as-is (already structured, not chunked)

    Returns:
        Summary dict: { source: chunk_count, "total": total_count }
    """
    logger.info("Starting preprocessing pipeline")

    all_chunks: list[dict] = []
    summary: dict[str, int] = {}

    # ── 1. Wikipedia ──────────────────────────────────────────────────────────
    wiki_records = _load_json("wikipedia.json")
    wiki_chunks  = process_records(wiki_records, default_source="wikipedia")
    all_chunks.extend(wiki_chunks)
    summary["wikipedia"] = len(wiki_chunks)

    # ── 2. Official Rules Pages ───────────────────────────────────────────────
    rules_records = _load_json("official_rules.json")
    rules_chunks  = process_records(rules_records, default_source="official_rules")
    all_chunks.extend(rules_chunks)
    summary["official_rules"] = len(rules_chunks)

    # ── 3. YouTube Transcripts ────────────────────────────────────────────────
    yt_records = _load_json("youtube_transcripts.json")
    yt_chunks  = process_records(yt_records, default_source="youtube_transcript")
    all_chunks.extend(yt_chunks)
    summary["youtube_transcripts"] = len(yt_chunks)

    # ── 4. Live Events — store as flat records, no chunking needed ────────────
    events_records = _load_json("live_events.json")
    # Events are already small structured dicts — wrap in a minimal text chunk
    event_chunks: list[dict] = []
    for i, ev in enumerate(events_records):
        # Build a natural-language representation for embedding
        text = (
            f"Event: {ev.get('title', 'Unknown')}. "
            f"Sport: {ev.get('sport', '')}. "
            f"Date: {ev.get('date', '')}. "
            f"Venue: {ev.get('venue', '')}. "
            f"Fighters: {ev.get('fighters', '')}. "
            f"Promotion: {ev.get('promotion', '')}."
        )
        text = clean_text(text)
        if len(text) >= MIN_CHUNK_LEN:
            event_chunks.append({
                "chunk_id":    f"live_event_{i:05d}",
                "text":        text,
                "sport":       ev.get("sport", "general"),
                "source":      "live_events",
                "doc_title":   ev.get("title", "Live Event"),
                "doc_url":     "",
                "chunk_index": 0,
                "total_chunks": 1,
            })

    all_chunks.extend(event_chunks)
    summary["live_events"] = len(event_chunks)

    # ── Save all chunks ───────────────────────────────────────────────────────
    save_chunks(all_chunks)

    total = sum(summary.values())
    summary["total"] = total
    logger.info(
        "Preprocessing complete",
        extra={"total_chunks": total, **summary},
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    setup_logging()
    run_preprocessing()
