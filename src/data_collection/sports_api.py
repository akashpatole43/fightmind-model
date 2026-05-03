"""
FightMind AI — Sports Events API Client
=========================================
Step 1.5 — Fetches live boxing, kickboxing, and karate events from
multiple sources with a 3-source fallback strategy.

Source priority (most reliable → least):
  1. TheSportsDB   (free, no key required beyond "1")
  2. API-Sports    (free tier, 100 req/day)
  3. Web scraping  (BoxingScene + GLORY kickboxing site, last resort)

Exception Handling Strategy:
  - Source 1 fails → log WARNING, try Source 2
  - Source 2 fails → log WARNING, try Source 3
  - Source 3 fails → log ERROR, return empty list (never crash the pipeline)
  - Each HTTP call: retried up to MAX_RETRIES with exponential back-off

Output: list of Event dicts saved to data/raw/live_events.json
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SPORTS_DB_API_KEY = os.getenv("SPORTS_DB_API_KEY", "1")       # "1" = free tier
API_SPORTS_KEY    = os.getenv("API_SPORTS_KEY", "")            # free tier at api-sports.io
MAX_RETRIES       = 3
BASE_BACKOFF      = 2.0                                         # seconds

# Path breakdown from src/data_collection/sports_api.py:
#   parents[0] = src/data_collection
#   parents[1] = src
#   parents[2] = fightmind-model  ← project root ✓
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """
    Normalized event record returned by all sources.
    All fields optional to accommodate partial data from different APIs.
    """
    title:      str                   # e.g. "Fury vs Usyk 2"
    sport:      str                   # "boxing" | "kickboxing" | "karate"
    date:       str                   # ISO 8601 date string "YYYY-MM-DD"
    time:       Optional[str] = None  # local time string, e.g. "20:00"
    venue:      Optional[str] = None  # e.g. "Kingdom Arena, Riyadh"
    fighters:   Optional[str] = None  # e.g. "Tyson Fury vs Oleksandr Usyk"
    promotion:  Optional[str] = None  # e.g. "Top Rank", "K-1"
    status:     Optional[str] = None  # "scheduled" | "live" | "finished"
    source:     str = "unknown"       # which API/scraper provided this


# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP Session
# ─────────────────────────────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "FightMindBot/1.0 (educational AI project)",
})


def _get_with_retry(url: str, headers: Optional[dict] = None, params: Optional[dict] = None) -> Optional[requests.Response]:
    """
    GET a URL with exponential back-off retry logic.

    Retry policy:
        Attempt 1: immediately
        Attempt 2: wait 2s
        Attempt 3: wait 4s
        After 3 failures: return None

    Returns:
        requests.Response on success, None on all retries exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _SESSION.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            return resp

        except requests.exceptions.HTTPError as exc:
            # 4xx — no point retrying (e.g. 401 Unauthorized, 429 Rate Limited)
            logger.warning(
                "HTTP error from API, not retrying",
                extra={"url": url, "status_code": exc.response.status_code if exc.response else "?", "attempt": attempt},
            )
            return None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            backoff = BASE_BACKOFF ** attempt
            logger.warning(
                "Network error reaching API, retrying",
                extra={"url": url, "attempt": attempt, "max_retries": MAX_RETRIES, "backoff_s": backoff},
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)

        except Exception as exc:
            logger.error("Unexpected error during HTTP request", exc_info=exc, extra={"url": url})
            return None

    logger.error("All retry attempts exhausted", extra={"url": url, "max_retries": MAX_RETRIES})
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — TheSportsDB (free, no sign-up needed beyond key "1")
# ─────────────────────────────────────────────────────────────────────────────

# TheSportsDB sport IDs for our three sports
_THESPORTSDB_SPORTS = {
    "boxing":     "Boxing",
    "kickboxing": "Kickboxing",
    "karate":     "Karate",
}

_THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json"


def _fetch_thesportsdb_events(target_date: Optional[str] = None) -> list[Event]:
    """
    Fetch upcoming events for boxing, kickboxing, and karate from TheSportsDB.

    Args:
        target_date: ISO date string "YYYY-MM-DD". Defaults to today.

    Returns:
        List of normalized Event objects.
    """
    target_date = target_date or date.today().isoformat()
    results: list[Event] = []

    logger.info("TheSportsDB: fetching events", extra={"date": target_date})

    for sport_key, sport_name in _THESPORTSDB_SPORTS.items():
        url = f"{_THESPORTSDB_BASE}/{SPORTS_DB_API_KEY}/eventsday.php"
        resp = _get_with_retry(url, params={"d": target_date, "s": sport_name})

        if not resp:
            logger.warning("TheSportsDB: no response for sport", extra={"sport": sport_key})
            continue

        try:
            data = resp.json()
            events_raw = data.get("events") or []

            if not events_raw:
                logger.debug("TheSportsDB: no events found for sport on date", extra={"sport": sport_key, "date": target_date})
                continue

            for ev in events_raw:
                event = Event(
                    title=ev.get("strEvent", "Unknown Event"),
                    sport=sport_key,
                    date=ev.get("dateEvent", target_date),
                    time=ev.get("strTime"),
                    venue=ev.get("strVenue"),
                    fighters=ev.get("strEvent"),         # TheSportsDB uses event name as fighter matchup
                    promotion=ev.get("strLeague"),
                    status=ev.get("strStatus", "scheduled"),
                    source="thesportsdb",
                )
                results.append(event)

            logger.debug(
                "TheSportsDB: events collected for sport",
                extra={"sport": sport_key, "count": len(events_raw)},
            )

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error(
                "TheSportsDB: failed to parse response",
                exc_info=exc,
                extra={"sport": sport_key},
            )

    logger.info("TheSportsDB: fetch complete", extra={"total_events": len(results)})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — API-Sports (free tier: 100 calls/day)
# ─────────────────────────────────────────────────────────────────────────────

_API_SPORTS_BASE = "https://v1.boxing.api-sports.io"


def _fetch_api_sports_events(target_date: Optional[str] = None) -> list[Event]:
    """
    Fetch boxing events from API-Sports (free tier).
    Only boxing is available on the free tier; kickboxing/karate not supported.

    Args:
        target_date: ISO date string "YYYY-MM-DD". Defaults to today.

    Returns:
        List of normalized Event objects.
    """
    if not API_SPORTS_KEY:
        logger.warning("API-Sports: no API key configured (API_SPORTS_KEY), skipping source 2")
        return []

    target_date = target_date or date.today().isoformat()
    results: list[Event] = []

    logger.info("API-Sports: fetching boxing events", extra={"date": target_date})

    headers = {
        "x-rapidapi-host": "v1.boxing.api-sports.io",
        "x-rapidapi-key": API_SPORTS_KEY,
    }

    resp = _get_with_retry(
        f"{_API_SPORTS_BASE}/fights",
        headers=headers,
        params={"date": target_date},
    )

    if not resp:
        logger.warning("API-Sports: no response received")
        return []

    try:
        data = resp.json()
        fights_raw = data.get("response") or []

        for fight in fights_raw:
            boxer1 = fight.get("boxer1", {}).get("name", "Unknown")
            boxer2 = fight.get("boxer2", {}).get("name", "Unknown")
            event = Event(
                title=f"{boxer1} vs {boxer2}",
                sport="boxing",
                date=fight.get("date", target_date),
                time=fight.get("time"),
                venue=fight.get("location", {}).get("venue"),
                fighters=f"{boxer1} vs {boxer2}",
                promotion=fight.get("promotion", {}).get("name"),
                status=fight.get("status", "scheduled"),
                source="api_sports",
            )
            results.append(event)

        logger.info("API-Sports: fetch complete", extra={"total_events": len(results)})

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("API-Sports: failed to parse response", exc_info=exc)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — Web Scraping (last resort)
# ─────────────────────────────────────────────────────────────────────────────

_SCRAPE_SOURCES = [
    {
        "name":    "BoxingScene Schedule",
        "url":     "https://www.boxingscene.com/schedule",
        "sport":   "boxing",
        "tag":     "table",
        "row_tag": "tr",
    },
    {
        "name":  "GLORY Kickboxing Events",
        "url":   "https://www.glorykickboxing.com/events",
        "sport": "kickboxing",
        "tag":   "div",
        "class_": "event-list",
    },
]


def _fetch_scraped_events(target_date: Optional[str] = None) -> list[Event]:
    """
    Scrape event listings from boxing/kickboxing sites as a last-resort fallback.
    Returns partial data — typically just title, date, and sport.

    Args:
        target_date: ISO date string "YYYY-MM-DD". Used for filtering if available.

    Returns:
        List of normalized Event objects (may have fewer fields than API sources).
    """
    target_date = target_date or date.today().isoformat()
    results: list[Event] = []

    logger.info("Web scraping: fetching events as fallback", extra={"date": target_date})

    for source in _SCRAPE_SOURCES:
        logger.debug("Web scraping: fetching source", extra={"source_name": source["name"], "url": source["url"]})

        resp = _get_with_retry(source["url"])
        if not resp:
            logger.warning("Web scraping: no response from source", extra={"source_name": source["name"]})
            continue

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Try to find event containers — simple heuristic: look for rows or event divs
            containers = []
            if "class_" in source:
                container = soup.find(source["tag"], class_=source["class_"])
                if container:
                    containers = container.find_all(source.get("row_tag", "div"))
            else:
                container = soup.find(source["tag"])
                if container:
                    containers = container.find_all(source.get("row_tag", "tr"))

            for item in containers[:20]:   # cap at 20 to avoid noise
                text = item.get_text(separator=" ", strip=True)
                if not text or len(text) < 5:
                    continue

                # Create a minimal event from whatever text we can extract
                event = Event(
                    title=text[:120],           # truncate long strings
                    sport=source["sport"],
                    date=target_date,           # we don't parse dates from HTML in fallback
                    source="web_scrape",
                )
                results.append(event)

            logger.debug(
                "Web scraping: collected events from source",
                extra={"source_name": source["name"], "count": len(containers)},
            )

        except Exception as exc:
            logger.error(
                "Web scraping: parse error for source",
                exc_info=exc,
                extra={"source_name": source["name"]},
            )

        time.sleep(1.0)   # polite delay between scraped sites

    logger.info("Web scraping: fetch complete", extra={"total_events": len(results)})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API — 3-Source Fallback Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def fetch_events(target_date: Optional[str] = None) -> list[dict]:
    """
    Fetch today's (or a given date's) martial arts events using a
    3-source fallback strategy:
        Source 1: TheSportsDB  (primary — always tried first)
        Source 2: API-Sports   (fallback — tried if Source 1 empty)
        Source 3: Web scraping (last resort — only if both APIs empty)

    Each source failure is logged at WARNING level, not raised,
    so the pipeline is never blocked by a single event API outage.

    Args:
        target_date: ISO date string "YYYY-MM-DD". Defaults to today.

    Returns:
        List of event dicts (serializable). Empty list if all sources fail.
    """
    target_date = target_date or date.today().isoformat()
    logger.info("Fetching events with 3-source fallback", extra={"date": target_date})

    # ── Source 1: TheSportsDB ─────────────────────────────────────────────────
    try:
        events = _fetch_thesportsdb_events(target_date)
    except Exception as exc:
        logger.warning("Source 1 (TheSportsDB) raised an exception, trying Source 2", exc_info=exc)
        events = []

    if events:
        logger.info("Events sourced from TheSportsDB", extra={"count": len(events)})
        return [asdict(e) for e in events]

    logger.warning("Source 1 (TheSportsDB) returned no events, trying Source 2 (API-Sports)")

    # ── Source 2: API-Sports ───────────────────────────────────────────────────
    try:
        events = _fetch_api_sports_events(target_date)
    except Exception as exc:
        logger.warning("Source 2 (API-Sports) raised an exception, trying Source 3", exc_info=exc)
        events = []

    if events:
        logger.info("Events sourced from API-Sports", extra={"count": len(events)})
        return [asdict(e) for e in events]

    logger.warning("Source 2 (API-Sports) returned no events, trying Source 3 (web scraping)")

    # ── Source 3: Web scraping ────────────────────────────────────────────────
    try:
        events = _fetch_scraped_events(target_date)
    except Exception as exc:
        logger.error("Source 3 (web scraping) raised an exception", exc_info=exc)
        events = []

    if events:
        logger.info("Events sourced from web scraping", extra={"count": len(events)})
    else:
        logger.error(
            "All 3 event sources returned no data",
            extra={"date": target_date},
        )

    return [asdict(e) for e in events]


def save_events(events: list[dict], filename: str = "live_events.json") -> Path:
    """
    Save fetched events to data/raw/<filename>.

    Raises:
        IOError / PermissionError: logged at CRITICAL and re-raised.
    """
    path = RAW_DIR / filename
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        logger.info(
            "Events saved to file",
            extra={"file_name": filename, "record_count": len(events), "path": str(path)},
        )
    except (IOError, PermissionError) as exc:
        logger.critical(
            "Failed to write events file — data will be lost",
            exc_info=exc,
            extra={"file_name": filename, "path": str(path)},
        )
        raise
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main — run directly for a quick fetch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    setup_logging()

    today = date.today().isoformat()
    logger.info("Running sports_api.py standalone", extra={"date": today})

    events = fetch_events(today)
    save_events(events)
    logger.info("Done", extra={"total_events": len(events)})
