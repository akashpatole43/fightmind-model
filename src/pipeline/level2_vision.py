"""
FightMind AI — Level 2: Vision Processing
=========================================
Step 1.11 — Uses Gemini 2.0 Flash to "see" images and translate them
into a text query that can be passed down to the RAG database.

If the user uploads an image, the Java backend saves it to Cloudinary
and sends us the URL. We download it into memory and feed it to Gemini
using Structured Outputs.
"""

import io
import os
from typing import List

import httpx
from google import genai
from PIL import Image
from pydantic import BaseModel, Field

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Structured Output Schemas (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class VisionResult(BaseModel):
    description: str = Field(
        description="A detailed description of the combat sport techniques shown in the image."
    )
    extracted_techniques: List[str] = Field(
        description="A list of specific techniques or concepts identified (e.g., 'high guard', 'roundhouse kick', 'southpaw stance').",
        default_factory=list
    )
    confidence: float = Field(
        description="Confidence score (0.0 to 1.0) that this image actually portrays boxing, kickboxing, or karate."
    )

class _GeminiVisionResult(BaseModel):
    """Internal model for the API call to bypass Pydantic $defs bugs."""
    description: str
    extracted_techniques: List[str]
    confidence: float


# ─────────────────────────────────────────────────────────────────────────────
# System Instruction
# ─────────────────────────────────────────────────────────────────────────────

VISION_PROMPT = """
You are a world-class martial arts analyst specializing in Boxing, Kickboxing, and Karate.

Analyze the provided image and describe what is happening.
Focus strictly on:
1. The stance, guard, and posture of the fighters.
2. The specific strikes (punches, kicks), defensive maneuvers (slips, blocks), or clinches being executed.
3. The specific sport, if identifiable.

If the image is completely unrelated to combat sports (e.g., a picture of a dog or a landscape), 
set confidence to 0.0 and describe what it is briefly.

Respond strictly in the provided JSON schema.
"""


def _download_image(image_url: str) -> Image.Image:
    """Download an image from a URL and convert it to a PIL Image."""
    logger.debug("Downloading image", extra={"url": image_url})
    
    headers = {
        "User-Agent": "FightMind-Bot/1.0 (https://github.com/fightmind; admin@example.com)"
    }
    with httpx.Client(timeout=10.0, headers=headers) as client:
        response = client.get(image_url)
        response.raise_for_status()
        
    return Image.open(io.BytesIO(response.content))


def analyze_image(
    image_url: str,
    user_prompt: str | None = None
) -> VisionResult:
    """
    Downloads the image and uses Gemini 2.0 Flash to extract martial arts techniques.

    Args:
        image_url: URL of the image to analyze (e.g., Cloudinary URL).
        user_prompt: Optional context from the user (e.g., "what's wrong with my guard here?").

    Returns:
        VisionResult containing description, techniques, and confidence.
    """
    if not image_url or not image_url.strip():
        logger.warning("Empty image URL passed to analyze_image")
        return VisionResult(
            description="",
            extracted_techniques=[],
            confidence=0.0
        )

    # 1. Download image
    try:
        pil_image = _download_image(image_url)
    except Exception as exc:
        logger.error("Failed to download or parse image", exc_info=exc, extra={"url": image_url})
        return VisionResult(
            description=f"Error analyzing image: failed to download from {image_url}",
            extracted_techniques=[],
            confidence=0.0
        )

    # 2. Build Gemini contents (Prompt + Image + User Text)
    contents = [VISION_PROMPT, pil_image]
    if user_prompt and user_prompt.strip():
        contents.append(f"User's question about this image: {user_prompt.strip()}")

    # 3. Call Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=contents,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_GeminiVisionResult,
                temperature=0.0
            ),
        )
        
        parsed: _GeminiVisionResult = response.parsed
        
        result = VisionResult(
            description=parsed.description,
            extracted_techniques=parsed.extracted_techniques,
            confidence=parsed.confidence
        )
        
        logger.info(
            "Image analyzed successfully",
            extra={
                "url": image_url,
                "confidence": result.confidence,
                "techniques": result.extracted_techniques
            }
        )
        return result

    except Exception as exc:
        logger.error("Failed to analyze image with Gemini", exc_info=exc, extra={"url": image_url})
        return VisionResult(
            description="Analysis failed due to an AI service error.",
            extracted_techniques=[],
            confidence=0.0
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.core.logging_config import setup_logging
    
    load_dotenv()
    setup_logging()
    
    import time
    print("Waiting 60 seconds to clear Gemini Free Tier quotas...")
    time.sleep(60)
    
    # Needs GEMINI_API_KEY + Internet connection
    # Using a known raw GitHub image of a martial artist to guarantee 200 OK
    test_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/107.png"  # Hitmonchan (Boxing Pokemon) as a fun, guaranteed-to-work test
    
    print("\n--- Testing Vision Analysis ---")
    print(f"URL: {test_url}")
    
    res = analyze_image(image_url=test_url, user_prompt="What stance is this fighter using?")
    
    print(f"\nDescription: {res.description}")
    print(f"Techniques:  {res.extracted_techniques}")
    print(f"Confidence:  {res.confidence}")
