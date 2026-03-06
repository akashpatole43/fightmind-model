"""
FightMind AI — API Health Tests
=================================
Tests /health and /pipeline/process endpoints.
"""


def test_health_endpoint(client):
    """Health check must return 200 and status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "fightmind-model"


def test_pipeline_stub_endpoint(client):
    """Pipeline endpoint must return 200 with an answer field."""
    payload = {
        "text": "What is a jab?",
        "user_id": "test-user",
        "skill_level": "beginner",
    }
    response = client.post("/pipeline/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence" in data
