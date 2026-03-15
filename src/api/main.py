"""
FightMind AI — Core API Server
==============================
Step 1.17 — Exposes the Python AI Pipeline via REST API.
This is the entry point that the Java Spring Boot Backend will communicate with.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.logging_config import get_logger, setup_logging
from src.api.schemas import ChatRequest
from src.pipeline_runner import run_pipeline, ChatbotResponse

# Initialize logging before booting the app
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FightMind AI Python Backend",
    description="The internal AI processing engine that classifies intents, searches ChromaDB, and generates LLM responses.",
    version="1.0.0",
)

# Apply CORS middleware (Allow Java Backend to call us)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev, ideally change to localhost:8080 in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check():
    """Simple health probe for Docker / CI/CD monitors."""
    return {"status": "ok", "service": "fightmind-ai-python"}


@app.post("/api/v1/chat", response_model=ChatbotResponse, tags=["AI Pipeline"])
async def process_chat(request: ChatRequest) -> ChatbotResponse:
    """
    Main entry point for incoming chat messages.
    Invokes the Phase 1D Pipeline (Levels 1 through 6).
    """
    logger.info("Received /api/v1/chat request", extra={"query_length": len(request.query)})
    
    try:
        # Run the synchronous pipeline 
        # (In the future, for high concurrency, we might want run_in_threadpool)
        response: ChatbotResponse = run_pipeline(
            query=request.query,
            image_url=request.image_url
        )
        return response
        
    except Exception as exc:
        logger.error("Unhandled exception escaping pipeline_runner", exc_info=exc)
        raise HTTPException(
            status_code=500,
            detail="FightMind AI encountered a critical error while processing the request."
        )

if __name__ == "__main__":
    import uvicorn
    # If run directly via python src/api/main.py
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
