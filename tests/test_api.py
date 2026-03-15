"""
Tests for the FastAPI Endpoints.
Ensures routing and serialization work as expected.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.pipeline_runner import ChatbotResponse

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "fightmind-ai-python"}

@patch("src.api.main.run_pipeline")
def test_chat_endpoint_success(mock_run_pipeline):
    """Verifies the /api/v1/chat endpoint correctly parses requests and serializes responses."""
    
    # Mock the internal pipeline so we don't hit Gemini
    mock_run_pipeline.return_value = ChatbotResponse(
        query="How to punch?",
        answer="Punching is fun.",
        confidence_score=0.99,
        detected_intent="TECHNIQUE",
        detected_sport="BOXING",
        user_skill_level="BEGINNER",
        used_vision=False,
        used_rag=True,
        used_live_events=False,
        hallucination_flag=False,
        fallback_engaged=False
    )
    
    payload = {
        "query": "How to punch?"
    }
    
    response = client.post("/api/v1/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["query"] == "How to punch?"
    assert data["answer"] == "Punching is fun."
    assert data["user_skill_level"] == "BEGINNER"
    assert data["used_rag"] is True
    
    mock_run_pipeline.assert_called_once_with(
        query="How to punch?",
        image_url=None
    )

def test_chat_endpoint_validation_error():
    """Verifies Pydantic catches missing required fields (query)."""
    payload = {
        # missing query
        "image_url": "http://image.com"
    }
    
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422 # Unprocessable Entity (FastAPI standard)
