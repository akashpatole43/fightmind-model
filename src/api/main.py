"""
FightMind AI — FastAPI Entry Point
====================================
Phase 1E, Step 1.17–1.18

Handles:
  - Application startup / shutdown lifecycle
  - Request/response logging middleware (every request logged with duration)
  - Global exception handler (unhandled errors → structured 500 response)
  - CORS for React frontend
  - /health and /pipeline/process endpoints
"""

import time
import uuid
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.logging_config import setup_logging, get_logger

# ── Bootstrap logging FIRST — before any other imports that log ───────────────
setup_logging()
logger = get_logger("fightmind.api")


# ─────────────────────────────────────────────────────────────────────────────
# Application Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs at startup and shutdown.
    Add resource initialisation (ChromaDB, model loading) here in later steps.
    """
    logger.info(
        "FightMind AI service starting up",
        extra={"env": os.getenv("ENV", "development"), "version": "1.0.0"},
    )
    yield   # Application runs here
    logger.info("FightMind AI service shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# App Initialisation
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FightMind AI — Model Service",
    description=(
        "6-Level AI Pipeline for Boxing, Kickboxing & Karate chatbot.\n\n"
        "Endpoints:\n"
        "- `GET /health` — liveness probe (Docker, UptimeRobot)\n"
        "- `POST /pipeline/process` — main AI pipeline entry point"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────

_allowed_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000",  # React dev servers
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Middleware — Log Every Request
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Assigns a unique request_id to every request and logs:
      - Incoming: method, path, client IP
      - Outgoing: status code, duration in ms

    The request_id threads through all log lines for that request,
    making it easy to trace a single request across log lines.
    """
    request_id = str(uuid.uuid4())[:8]   # Short 8-char ID, e.g. "a3f2c1b0"
    start_time = time.monotonic()

    logger.info(
        "→ Incoming request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        },
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        # Unexpected error escaped all other handlers — log it with full trace
        duration_ms = round((time.monotonic() - start_time) * 1000, 2)
        logger.critical(
            "Unhandled exception in request",
            exc_info=exc,
            extra={"request_id": request_id, "duration_ms": duration_ms},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
                "request_id": request_id,
            },
        )

    duration_ms = round((time.monotonic() - start_time) * 1000, 2)

    log_fn = logger.warning if response.status_code >= 400 else logger.info
    log_fn(
        "← Response sent",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    # Pass request_id back to client for support/debugging
    response.headers["X-Request-ID"] = request_id
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Global Exception Handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles FastAPI HTTPExceptions (404, 422, etc.) with structured JSON.
    Logs at WARNING level since these are expected client errors.
    """
    logger.warning(
        "HTTP exception raised",
        extra={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "status_code": exc.status_code,
            "message": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for any unhandled Python exceptions.
    Logs at ERROR level with full traceback for post-mortem debugging.
    """
    logger.error(
        "Unhandled application exception",
        exc_info=exc,
        extra={"path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again.",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Liveness probe.
    Polled by:
      - Docker Compose healthcheck (every 30s)
      - UptimeRobot in production (every 14 min to prevent Render sleep)

    Returns 200 when the service is ready to accept requests.
    """
    return {
        "status": "ok",
        "service": "fightmind-model",
        "version": "1.0.0",
        "environment": os.getenv("ENV", "development"),
    }


@app.post("/pipeline/process", tags=["Pipeline"])
async def process_query(payload: dict):
    """
    Main 6-level AI pipeline endpoint.

    Expected request body:
    {
        "text":        "What is a jab?",          # optional if image_url provided
        "image_url":   "https://...",              # optional if text provided
        "user_id":     "user-uuid-here",           # required
        "skill_level": "beginner"                  # beginner | intermediate | advanced
    }

    Response:
    {
        "answer":       "A jab is a quick straight punch...",
        "confidence":   0.92,
        "sources":      [{"title": "Boxing", "url": "..."}],
        "level_outputs": { "L1": {...}, "L3": {...} }
    }

    Full pipeline implementation is added step-by-step in Phase 1D (Steps 1.10–1.16).
    """
    user_id = payload.get("user_id", "anonymous")
    has_text = bool(payload.get("text"))
    has_image = bool(payload.get("image_url"))

    # Validate: at least one input must be provided
    if not has_text and not has_image:
        logger.warning(
            "Pipeline request rejected — no text or image provided",
            extra={"user_id": user_id},
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Provide at least one of: 'text' or 'image_url'.",
            },
        )

    logger.info(
        "Pipeline request received",
        extra={
            "user_id": user_id,
            "has_text": has_text,
            "has_image": has_image,
            "skill_level": payload.get("skill_level", "unknown"),
        },
    )

    # ── Placeholder — replaced by pipeline_runner.py in Step 1.16 ────────────
    return {
        "answer": "FightMind AI is online. Full pipeline coming in Phase 1D.",
        "confidence": 1.0,
        "sources": [],
        "level_outputs": {},
    }
