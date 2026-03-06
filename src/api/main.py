"""
FightMind AI — FastAPI Entry Point
===================================
Phase 1E, Step 1.17–1.18
Skeleton version: boots, serves /health, ready for pipeline integration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fightmind.api")

# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FightMind AI — Model Service",
    description="6-Level AI Pipeline for Boxing, Kickboxing & Karate chatbot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened in production via env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Endpoint (Step 1.18) ───────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint polled by:
    - Docker Compose healthcheck
    - UptimeRobot (production keep-alive)
    """
    return {
        "status": "ok",
        "service": "fightmind-model",
        "version": "1.0.0",
        "environment": os.getenv("ENV", "development"),
    }


# ── Pipeline Endpoint (Step 1.17) — Stub, filled in Phase 1E ─────────────────
@app.post("/pipeline/process", tags=["Pipeline"])
async def process_query(payload: dict):
    """
    Main pipeline endpoint.
    Accepts: { "text": "...", "image_url": "...", "user_id": "...", "skill_level": "beginner" }
    Returns: { "answer": "...", "confidence": 0.95, "sources": [...] }

    Full implementation built step-by-step in Phase 1D–1E.
    """
    logger.info(f"Received query from user_id={payload.get('user_id', 'anonymous')}")

    # Placeholder response — replaced by pipeline_runner in Step 1.16
    return {
        "answer": "FightMind AI is online. Pipeline implementation in progress.",
        "confidence": 1.0,
        "sources": [],
        "level_outputs": {},
    }
