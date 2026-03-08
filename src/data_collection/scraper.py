"""
FightMind AI — Domain Knowledge Scraper
========================================
Step 1.4 — Collects raw text from multiple sources and saves to data/raw/.

Sources:
  1. Wikipedia        — fighter profiles, rules, history, techniques
  2. Official Sites   — WBC, WBA, IBF, WBO rules pages (BeautifulSoup)
  3. YouTube          — technique tutorial transcripts

Output: JSON files in data/raw/ — one per source category.

Exception Handling Strategy:
  - Per-topic   : Log WARNING and continue (one failure ≠ abort all scraping)
  - Per-source  : Log ERROR and continue (scraper failure ≠ abort other scrapers)
  - HTTP        : Retry up to MAX_RETRIES with exponential back-off before giving up
  - Critical IO : Log CRITICAL and re-raise (cannot write output files)
"""

import json
import logging
import time
import re
from pathlib import Path
from typing import Optional

import requests
import wikipedia
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from tqdm import tqdm

from src.core.logging_config import get_logger

# ── Logger — name follows module path for easy filtering ─────────────────────
logger = get_logger(__name__)   # resolves to "src.data_collection.scraper"

# ── Retry config for HTTP requests ───────────────────────────────────────
MAX_RETRIES = 3          # max attempts per URL
BASE_BACKOFF = 2.0       # seconds (doubles each retry: 2 → 4 → 8)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Path breakdown from src/data_collection/scraper.py:
#   parents[0] = src/data_collection
#   parents[1] = src
#   parents[2] = fightmind-model  ← project root ✓
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── HTTP Session (shared, with retries) ───────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "FightMindBot/1.0 (educational AI project)"
    )
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. WIKIPEDIA SCRAPER
# ─────────────────────────────────────────────────────────────────────────────

# Topics to collect — covers boxing, kickboxing, karate
WIKIPEDIA_TOPICS = [

    # ════════════════════════════════════════════════════════════════════════
    # BOXING — COMPLETE COVERAGE
    # ════════════════════════════════════════════════════════════════════════

    # ── Forms & Rulesets ─────────────────────────────────────────────────────
    "Boxing",
    "Amateur boxing",
    "Professional boxing",
    "Olympic boxing",
    "Bare-knuckle boxing",
    "White-collar boxing",
    "Exhibition boxing",
    "Prizefighting",
    "Queensberry Rules",
    "London Prize Ring rules",

    # ── Boxing Organisations ─────────────────────────────────────────────────
    "World Boxing Council",
    "World Boxing Association",
    "International Boxing Federation",
    "World Boxing Organization",
    "International Boxing Association",
    "World Boxing Super Series",

    # ── Weight Classes ────────────────────────────────────────────────────────
    "Boxing weight classes",
    "Strawweight",
    "Light flyweight",
    "Flyweight (boxing)",
    "Super flyweight",
    "Bantamweight",
    "Super bantamweight",
    "Featherweight (boxing)",
    "Super featherweight",
    "Lightweight (boxing)",
    "Super lightweight",
    "Welterweight (boxing)",
    "Super welterweight",
    "Middleweight (boxing)",
    "Super middleweight",
    "Light heavyweight (boxing)",
    "Cruiserweight (boxing)",
    "Heavyweight (boxing)",

    # ── Fighting Styles ───────────────────────────────────────────────────────
    "Swarmer (boxing)",
    "Out-boxer",
    "Slugger (boxing)",
    "Boxer-puncher",
    "Southpaw stance",
    "Orthodox stance",
    "Switch-hitter (boxing)",

    # ── Offensive Techniques ─────────────────────────────────────────────────
    "Jab (boxing)",
    "Cross (boxing)",
    "Hook (boxing)",
    "Uppercut",
    "Body shot (boxing)",
    "Overhand (boxing)",
    "Bolo punch",
    "Check hook",
    "Boxing combinations",

    # ── Defensive Techniques ─────────────────────────────────────────────────
    "Guard (boxing)",
    "Bob and weave",
    "Slip (boxing)",
    "Parry (boxing)",
    "Clinch (boxing)",
    "Shoulder roll",
    "Philly shell",
    "Peek-a-boo style",
    "Footwork in boxing",

    # ── Match Outcomes & Scoring ─────────────────────────────────────────────
    "Knockout",
    "Technical knockout",
    "Split decision",
    "Unanimous decision",
    "Majority decision",
    "Draw (boxing)",
    "No contest",
    "Disqualification (boxing)",
    "Boxing scoring",
    "Three-knockdown rule",

    # ── Training & Equipment ─────────────────────────────────────────────────
    "Boxing training",
    "Sparring",
    "Speed bag",
    "Heavy bag",
    "Boxing gloves",
    "Mouthguard",
    "Headgear (boxing)",
    "Hand wrapping",
    "Boxing ring",
    "Cornerman",
    "Trainer (boxing)",
    "Cutman",

    # ── Famous Boxers ─────────────────────────────────────────────────────────
    "Muhammad Ali",
    "Mike Tyson",
    "Floyd Mayweather Jr.",
    "Manny Pacquiao",
    "Sugar Ray Leonard",
    "Joe Frazier",
    "George Foreman",
    "Lennox Lewis",
    "Canelo Álvarez",
    "Tyson Fury",
    "Anthony Joshua",
    "Oleksandr Usyk",
    "Joe Louis",
    "Rocky Marciano",
    "Sugar Ray Robinson",
    "Oscar De La Hoya",
    "Bernard Hopkins",
    "Roy Jones Jr.",
    "Evander Holyfield",
    "Wladimir Klitschko",

    # ── Famous Fights / History ───────────────────────────────────────────────
    "Thrilla in Manila",
    "Rumble in the Jungle",
    "Boxing at the Summer Olympics",

    # ════════════════════════════════════════════════════════════════════════
    # KICKBOXING — COMPLETE COVERAGE
    # ════════════════════════════════════════════════════════════════════════

    # ── Forms & Rulesets ─────────────────────────────────────────────────────
    "Kickboxing",
    "American kickboxing",
    "Japanese kickboxing",
    "Dutch kickboxing",
    "Low kick kickboxing",
    "Full contact karate",
    "Semi-contact karate",
    "Continuous sparring (martial arts)",
    "Muay Thai",
    "Savate",

    # ── Organisations & Promotions ────────────────────────────────────────────
    "K-1",
    "Glory (kickboxing)",
    "ONE Championship",
    "ISKA (International Sport Karate Association)",
    "World Association of Kickboxing Organizations",

    # ── Kickboxing Techniques ─────────────────────────────────────────────────
    "Roundhouse kick",
    "Front kick",
    "Side kick",
    "Back kick",
    "Spinning back kick",
    "Axe kick",
    "Crescent kick",
    "Low kick",
    "Teep",
    "Knee strike",
    "Elbow strike",
    "Spinning heel kick",

    # ── Kickboxing Stances & Footwork ─────────────────────────────────────────
    "Fighting stance",
    "Southpaw stance",

    # ── Famous Kickboxers ─────────────────────────────────────────────────────
    "Ernesto Hoost",
    "Peter Aerts",
    "Mirko Cro Cop",
    "Buakaw Banchamek",
    "Giorgio Petrosyan",
    "Rico Verhoeven",
    "Sitthichai Sitsongpeenong",
    "Robin van Roosmalen",

    # ════════════════════════════════════════════════════════════════════════
    # KARATE — COMPLETE COVERAGE
    # ════════════════════════════════════════════════════════════════════════

    # ── History & Overview ────────────────────────────────────────────────────
    "Karate",
    "History of karate",
    "Sport karate",
    "Karate at the Summer Olympics",
    "World Karate Federation",
    "Japan Karate Association",

    # ── Main Styles (Ryu-ha) ──────────────────────────────────────────────────
    "Shotokan",
    "Goju-ryu",
    "Wado-ryu",
    "Shito-ryu",
    "Kyokushin",
    "Uechi-ryu",
    "Shorin-ryu",
    "Shorei-ryu",
    "Isshin-ryu",
    "Budokan (martial art)",

    # ── Kumite Types ──────────────────────────────────────────────────────────
    "Kumite",
    "Ippon kumite",
    "Sanbon kumite",
    "Jiyu kumite",
    "Shiai kumite",

    # ── Kata ──────────────────────────────────────────────────────────────────
    "Kata (martial arts)",
    "Heian (kata)",
    "Tekki (kata)",
    "Bassai",
    "Kanku",
    "Hangetsu",
    "Jion (kata)",
    "Empi (kata)",
    "Gankaku",
    "Sanchin",

    # ── Fundamental Techniques ────────────────────────────────────────────────
    "Kihon",
    "Oi-zuki",
    "Gyaku-zuki",
    "Age-uke",
    "Soto-uke",
    "Gedan-barai",
    "Mae-geri",
    "Yoko-geri",
    "Mawashi-geri",
    "Ushiro-geri",

    # ── Stances ───────────────────────────────────────────────────────────────
    "Zenkutsu-dachi",
    "Kokutsu-dachi",
    "Kiba-dachi",
    "Musubi-dachi",

    # ── Dojo & Culture ────────────────────────────────────────────────────────
    "Dojo",
    "Gi (uniform)",
    "Belt (martial arts)",
    "Black belt (martial arts)",
    "Sensei",
    "Kyu (martial arts)",
    "Dan (rank)",
    "Bowing (martial arts)",
    "Dojo kun",
    "Bushido",

    # ── Famous Karate Practitioners ───────────────────────────────────────────
    "Gichin Funakoshi",
    "Masutatsu Oyama",
    "Hirokazu Kanazawa",
    "Antonio Diaz (karateka)",
    "Rafael Aghayev",
    "Sandra Sanchez",

    # ════════════════════════════════════════════════════════════════════════
    # GENERAL COMBAT SPORTS (shared context)
    # ════════════════════════════════════════════════════════════════════════
    "Combat sport",
    "Martial arts",
    "Mixed martial arts",
    "Self-defense",
    "Conditioning (exercise)",
    "Plyometrics",
    "Shadow boxing",
]


def scrape_wikipedia(topics: list[str] = WIKIPEDIA_TOPICS) -> list[dict]:
    """
    Fetch full Wikipedia article text for each topic.

    Returns:
        List of dicts: { title, text, url, source }

    Exception Handling:
        - DisambiguationError: tries the first suggested option
        - PageError: topic not found — skip and log WARNING
        - All other exceptions: log ERROR and skip (scraping is best-effort)
    """
    wikipedia.set_lang("en")
    results = []
    skipped = 0

    logger.info("Wikipedia scraper starting", extra={"topic_count": len(topics)})

    for topic in tqdm(topics, desc="Wikipedia"):
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            results.append({
                "title": page.title,
                "text": page.content,
                "url": page.url,
                "source": "wikipedia",
            })
            logger.debug("Wikipedia article collected", extra={"topic": topic, "chars": len(page.content)})
            time.sleep(0.3)   # polite delay — respect Wikipedia rate limits

        except wikipedia.DisambiguationError as e:
            # Disambiguation page: try the first suggested option
            logger.debug("Disambiguation for topic, trying first option", extra={"topic": topic, "option": e.options[0] if e.options else None})
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                results.append({
                    "title": page.title,
                    "text": page.content,
                    "url": page.url,
                    "source": "wikipedia",
                })
            except wikipedia.PageError:
                logger.warning("Disambiguation fallback also not found", extra={"topic": topic})
                skipped += 1
            except Exception as inner_exc:
                logger.warning("Disambiguation fallback failed", extra={"topic": topic, "error": str(inner_exc)})
                skipped += 1
            time.sleep(0.3)

        except wikipedia.PageError:
            # Topic simply does not exist on Wikipedia — safe to skip
            logger.warning("Wikipedia topic not found, skipping", extra={"topic": topic})
            skipped += 1

        except requests.exceptions.ConnectionError:
            # Network issue — log clearly and continue
            logger.error("Network error fetching Wikipedia article", extra={"topic": topic})
            skipped += 1

        except Exception as exc:
            # Unexpected error — log with traceback for debugging, then skip
            logger.error(
                "Unexpected error fetching Wikipedia article",
                exc_info=exc,
                extra={"topic": topic},
            )
            skipped += 1

    logger.info(
        "Wikipedia scraper complete",
        extra={"collected": len(results), "skipped": skipped},
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. OFFICIAL RULES SCRAPER (BeautifulSoup)
# ─────────────────────────────────────────────────────────────────────────────

# Pages known to be publicly accessible and stable
RULES_PAGES = [

    # ── Boxing ───────────────────────────────────────────────────────────────
    {
        "name": "WBC Rules",
        "url": "https://www.wbcboxing.com/en/rules/",
        "tag": "article",
    },
    {
        "name": "WBA Rules",
        "url": "https://www.wbaboxing.com/wba-rules",
        "tag": "div",
        "class_": "entry-content",
    },
    {
        "name": "Amateur Boxing (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Amateur_boxing",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "Professional Boxing (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Professional_boxing",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "Boxing Weight Classes (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Boxing_weight_classes",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "Queensberry Rules (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Queensberry_Rules",
        "tag": "div",
        "class_": "mw-parser-output",
    },

    # ── Kickboxing ───────────────────────────────────────────────────────────
    {
        "name": "Kickboxing Overview (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Kickboxing",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "American Kickboxing (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/American_kickboxing",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "K-1 Rules (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/K-1",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "Muay Thai (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Muay_Thai",
        "tag": "div",
        "class_": "mw-parser-output",
    },

    # ── Karate ───────────────────────────────────────────────────────────────
    {
        "name": "Sport Karate Rules (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Sport_karate",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "Karate Olympics (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Karate_at_the_2020_Summer_Olympics",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "Kyokushin Karate (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Kyokushin",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "Kumite Rules (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Kumite",
        "tag": "div",
        "class_": "mw-parser-output",
    },
    {
        "name": "Kata (Wikipedia)",
        "url": "https://en.wikipedia.org/wiki/Kata",
        "tag": "div",
        "class_": "mw-parser-output",
    },
]


def _fetch_with_retry(url: str, tag: str, class_: Optional[str] = None) -> Optional[str]:
    """
    Fetch a URL with exponential back-off retry logic.
    Extracts paragraph text from the specified HTML tag/class.

    Retry policy:
        - Attempt 1: immediately
        - Attempt 2: wait 2s
        - Attempt 3: wait 4s
        - After 3 failures: return None (caller logs and skips)

    Args:
        url:    Target URL to scrape
        tag:    HTML element to search within (e.g. "div", "article")
        class_: Optional CSS class to narrow the element search

    Returns:
        Extracted text string, or None if all retries failed
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=15)
            resp.raise_for_status()   # raises HTTPError for 4xx/5xx

            soup = BeautifulSoup(resp.text, "html.parser")
            container = soup.find(tag, class_=class_) if class_ else soup.find(tag)

            # Fallback: if specific tag not found, use entire body
            if not container:
                logger.debug("HTML tag not found in page, falling back to full body", extra={"url": url, "tag": tag})
                container = soup

            paragraphs = container.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return text if text else None

        except requests.exceptions.HTTPError as exc:
            # 4xx errors (forbidden, not found) — no point retrying
            logger.warning(
                "HTTP error fetching page, will not retry",
                extra={"url": url, "status_code": exc.response.status_code if exc.response else "unknown"},
            )
            return None

        except requests.exceptions.ConnectionError:
            backoff = BASE_BACKOFF ** attempt
            logger.warning(
                "Connection error fetching page, retrying",
                extra={"url": url, "attempt": attempt, "max_retries": MAX_RETRIES, "backoff_s": backoff},
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)

        except requests.exceptions.Timeout:
            backoff = BASE_BACKOFF ** attempt
            logger.warning(
                "Request timed out, retrying",
                extra={"url": url, "attempt": attempt, "backoff_s": backoff},
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)

        except Exception as exc:
            logger.error("Unexpected error parsing page", exc_info=exc, extra={"url": url})
            return None

    logger.error("All retry attempts exhausted for URL", extra={"url": url, "max_retries": MAX_RETRIES})
    return None


# Keep the old name as an alias for backwards compatibility in tests
_fetch_page_text = _fetch_with_retry


def scrape_rules_pages(pages: list[dict] = RULES_PAGES) -> list[dict]:
    """
    Scrape official boxing/kickboxing/karate rules and reference pages.

    Returns:
        List of dicts: { title, text, url, source }

    Exception Handling:
        - HTTP errors + timeouts: retried up to MAX_RETRIES by _fetch_with_retry
        - No text extracted: log WARNING and skip
    """
    results = []
    skipped = 0
    logger.info("Rules page scraper starting", extra={"page_count": len(pages)})

    for page in tqdm(pages, desc="Official pages"):
        logger.debug("Fetching rules page", extra={"page_name": page["name"], "url": page["url"]})

        text = _fetch_with_retry(
            url=page["url"],
            tag=page["tag"],
            class_=page.get("class_"),
        )

        if text:
            results.append({
                "title": page["name"],
                "text": text,
                "url": page["url"],
                "source": "official_rules",
            })
            logger.debug("Rules page collected", extra={"page_name": page["name"], "chars": len(text)})
        else:
            logger.warning("No text extracted from rules page, skipping", extra={"page_name": page["name"], "url": page["url"]})
            skipped += 1

        time.sleep(1.0)   # polite delay between pages

    logger.info(
        "Rules page scraper complete",
        extra={"collected": len(results), "skipped": skipped},
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. YOUTUBE TRANSCRIPT SCRAPER
# ─────────────────────────────────────────────────────────────────────────────

# Curated YouTube video IDs for martial arts technique tutorials
# Format: { "title": "...", "video_id": "..." }
YOUTUBE_VIDEOS = [

    # ── Boxing — Fundamentals ─────────────────────────────────────────────────
    {"title": "Boxing Basics - Stance and Footwork", "video_id": "IOe7T3xkbP8"},
    {"title": "How to Throw a Jab", "video_id": "OM4ZLXG-9wE"},
    {"title": "How to Throw a Cross, Hook, and Uppercut", "video_id": "9XJageRJBeA"},
    {"title": "Boxing Combinations for Beginners", "video_id": "iF5KJXeCjgs"},
    {"title": "How to Slip a Punch - Boxing Defense", "video_id": "HlXhHNOoAdk"},
    {"title": "Southpaw vs Orthodox Stance Explained", "video_id": "m9s0cP58eGo"},
    {"title": "Peek-a-Boo Boxing Style - Mike Tyson", "video_id": "lpTZUMIXnbg"},
    {"title": "Philly Shell Defensive Style Explained", "video_id": "zVBMEwsWvBM"},
    {"title": "Boxing Weight Classes Explained", "video_id": "7EVbTCzI5Dg"},
    {"title": "Amateur vs Professional Boxing Differences", "video_id": "mB2v0UjOGow"},

    # ── Kickboxing — All Forms ────────────────────────────────────────────────
    {"title": "Roundhouse Kick Tutorial for Beginners", "video_id": "e-2OQKM3KEI"},
    {"title": "Kickboxing Basics - Front Kick and Side Kick", "video_id": "YBquCNFH-lE"},
    {"title": "Low Kick Technique in Kickboxing", "video_id": "YzGLNZ9BKGA"},
    {"title": "K-1 Kickboxing Rules Explained", "video_id": "5g5d3oyAv2g"},
    {"title": "American Kickboxing vs Dutch Kickboxing", "video_id": "lpK3LRQA5OY"},
    {"title": "Point Fighting (Semi-Contact) Techniques", "video_id": "RYGJ4rNY8IY"},
    {"title": "Full Contact Kickboxing Basics", "video_id": "VBbFH3a5H2M"},
    {"title": "Muay Thai vs Kickboxing Key Differences", "video_id": "Y5hFNEPFRHQ"},
    {"title": "Spinning Back Kick Tutorial", "video_id": "NWa9d3VKd6E"},
    {"title": "Axe Kick and Crescent Kick Technique", "video_id": "M24Y1rEEeRE"},

    # ── Karate — All Styles & Types ───────────────────────────────────────────
    {"title": "Shotokan Karate Basics - Kihon", "video_id": "wWMb_AAOU0c"},
    {"title": "Karate Kata Tutorial - Heian Shodan", "video_id": "MIxFcIGOnWI"},
    {"title": "Heian Nidan Kata Step by Step", "video_id": "fgGVvqSBYDk"},
    {"title": "Kyokushin Karate Basics", "video_id": "5HBX3GBBnzI"},
    {"title": "Goju-Ryu Karate Sanchin Kata", "video_id": "Eo0TaJPNnxw"},
    {"title": "Karate Kumite Scoring Rules Explained", "video_id": "K_mCpmoxcV4"},
    {"title": "Sport Karate Point Fighting - Jiyu Kumite", "video_id": "M4p8H-8WJMI"},
    {"title": "Karate Belt System Explained", "video_id": "aH2BxzKtE8I"},
    {"title": "Wado-Ryu Karate Introduction", "video_id": "gzLNJVoiSq8"},
    {"title": "Karate Stances - Zenkutsu Kiba Kokutsu Dachi", "video_id": "DSMJ8KNtbNY"},
]


def scrape_youtube_transcripts(videos: list[dict] = YOUTUBE_VIDEOS) -> list[dict]:
    """
    Fetch English transcripts from YouTube videos.
    Prefers manual captions; falls back to auto-generated.

    Returns:
        List of dicts: { title, text, url, source }

    Exception Handling:
        - TranscriptsDisabled  : video has no captions — skip, log WARNING
        - NoTranscriptFound    : no English transcript — skip, log WARNING
        - All other exceptions : log ERROR and continue (best-effort)
    """
    results = []
    skipped = 0
    logger.info("YouTube scraper starting", extra={"video_count": len(videos)})

    for video in tqdm(videos, desc="YouTube"):
        vid_id = video["video_id"]
        url = f"https://www.youtube.com/watch?v={vid_id}"

        logger.debug("Fetching YouTube transcript", extra={"video_id": vid_id, "title": video["title"]})

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                vid_id,
                languages=["en", "en-US", "en-GB"],
            )
            # Merge all segments into a single clean text block
            text = " ".join(
                re.sub(r"\s+", " ", seg["text"].strip())
                for seg in transcript_list
                if seg["text"].strip()
            )
            if text:
                results.append({
                    "title": video["title"],
                    "text": text,
                    "url": url,
                    "source": "youtube_transcript",
                })
                logger.debug(
                    "YouTube transcript collected",
                    extra={"title": video["title"], "chars": len(text), "video_id": vid_id},
                )
            else:
                logger.warning("Transcript was empty after cleaning", extra={"video_id": vid_id})
                skipped += 1

        except TranscriptsDisabled:
            logger.warning("Transcripts disabled for video", extra={"video_id": vid_id, "title": video["title"]})
            skipped += 1

        except NoTranscriptFound:
            logger.warning("No English transcript found for video", extra={"video_id": vid_id, "title": video["title"]})
            skipped += 1

        except Exception as exc:
            logger.error(
                "Unexpected error fetching YouTube transcript",
                exc_info=exc,
                extra={"video_id": vid_id, "title": video["title"]},
            )
            skipped += 1

        time.sleep(0.5)   # polite delay

    logger.info(
        "YouTube scraper complete",
        extra={"collected": len(results), "skipped": skipped},
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(data: list[dict], filename: str) -> Path:
    """
    Save a list of records as JSON to data/raw/<filename>.

    Exception Handling:
        - IOError / PermissionError: logged at CRITICAL and re-raised.
          Output files are essential — a failure here means data is lost.
    """
    path = RAW_DIR / filename
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Data saved to file", extra={"file_name": filename, "record_count": len(data), "path": str(path)})
    except (IOError, PermissionError) as exc:
        logger.critical(
            "Failed to write output file — data will be lost",
            exc_info=exc,
            extra={"file_name": filename, "path": str(path)},
        )
        raise   # Re-raise: caller must know the save failed
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scrapers() -> dict[str, int]:
    """
    Orchestrate all scrapers in sequence and save results to data/raw/.

    Each scraper runs independently — a failure in one does NOT stop the others.
    Summary statistics are logged at INFO level at the end.

    Returns:
        dict mapping source name to number of documents collected, e.g.
        { "wikipedia": 170, "official_rules": 12, "youtube_transcripts": 25 }
    """
    summary: dict[str, int] = {}
    logger.info("Starting full data collection run")

    # 1. Wikipedia
    try:
        wiki_data = scrape_wikipedia()
        _save_json(wiki_data, "wikipedia.json")
        summary["wikipedia"] = len(wiki_data)
    except Exception as exc:
        logger.error("Wikipedia scraper run failed", exc_info=exc)
        summary["wikipedia"] = 0

    # 2. Official rules pages
    try:
        rules_data = scrape_rules_pages()
        _save_json(rules_data, "official_rules.json")
        summary["official_rules"] = len(rules_data)
    except Exception as exc:
        logger.error("Rules page scraper run failed", exc_info=exc)
        summary["official_rules"] = 0

    # 3. YouTube transcripts
    try:
        yt_data = scrape_youtube_transcripts()
        _save_json(yt_data, "youtube_transcripts.json")
        summary["youtube_transcripts"] = len(yt_data)
    except Exception as exc:
        logger.error("YouTube scraper run failed", exc_info=exc)
        summary["youtube_transcripts"] = 0

    total = sum(summary.values())
    logger.info(
        "Data collection run complete",
        extra={"total_documents": total, **summary},
    )
    return summary


if __name__ == "__main__":
    # When run directly, bootstrap logging in human-readable mode
    from src.core.logging_config import setup_logging
    setup_logging()
    run_all_scrapers()
