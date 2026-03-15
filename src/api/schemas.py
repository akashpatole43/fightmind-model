"""
FastAPI Pydantic Schemas
Defines incoming request structure for the FightMind AI Backend.
"""

from typing import Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        description="The text query from the user (e.g., 'How do I throw a jab?')",
        example="How do I defend a rear naked choke?"
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Optional URL to an image the user uploaded for technique analysis.",
        example="https://example.com/choke-position.jpg"
    )
