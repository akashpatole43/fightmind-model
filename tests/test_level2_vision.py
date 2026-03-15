"""
Offline tests for Level 2 Vision Processing (Step 1.11).

Mocks `httpx.Client.get` and `google.genai.Client` to simulate image downloads
and multimodal AI analysis without network IO or API keys.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.pipeline.level2_vision import VisionResult, analyze_image


@pytest.fixture
def mock_genai_client():
    """Mocks the Google GenAI client."""
    with patch("src.pipeline.level2_vision.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        mock_response = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        
        yield mock_client, mock_response


@pytest.fixture
def mock_httpx_get():
    """Mocks httpx.Client.get to avoid real HTTP requests."""
    with patch("src.pipeline.level2_vision.httpx.Client.get") as mock_get:
        # Create a tiny 1x1 black GIF image as fake bytes
        fake_image_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x01D\x00;"
        mock_response = MagicMock()
        mock_response.content = fake_image_bytes
        # Do not raise an exception on raise_for_status by default
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get


class TestAnalyzeImage:

    def test_empty_url_returns_gracefully(self):
        """Test that an empty URL skips processing and returns an empty VisionResult."""
        result = analyze_image("   \n")
        assert result.confidence == 0.0
        assert result.description == ""
        assert len(result.extracted_techniques) == 0

    def test_analyze_image_success(self, mock_httpx_get, mock_genai_client):
        """Happy path — simulates successful download and Gemini parsing."""
        mock_client, mock_response = mock_genai_client
        
        # Simulate gemini returning a parsed Pydantic object
        # Note: the script converts _GeminiVisionResult to VisionResult internally
        from src.pipeline.level2_vision import _GeminiVisionResult
        
        mock_response.parsed = _GeminiVisionResult(
            description="A fighter throwing a right cross.",
            extracted_techniques=["Right cross", "Orthodox stance"],
            confidence=0.98
        )
        
        result = analyze_image(
            image_url="http://fake.url/image.jpg",
            user_prompt="What punch is this?"
        )
        
        assert mock_httpx_get.called
        assert mock_client.models.generate_content.called
        
        assert result.description == "A fighter throwing a right cross."
        assert result.extracted_techniques == ["Right cross", "Orthodox stance"]
        assert result.confidence == 0.98
        
        # Verify the prompt was passed correctly
        kwargs = mock_client.models.generate_content.call_args.kwargs
        contents_list = kwargs["contents"]
        assert len(contents_list) == 3  # SYSTEM PROMPT + Image + User Prompt
        assert "User's question about this image: What punch is this?" in contents_list[2]

    def test_http_download_failure(self, mock_httpx_get, mock_genai_client):
        """Test that an invalid URL/failed download returns safely."""
        import httpx
        mock_httpx_get.side_effect = httpx.HTTPError("Network failed")
        
        result = analyze_image("http://bad.url/image.jpg")
        
        assert "Error analyzing image" in result.description
        assert result.confidence == 0.0
        
        # Should NOT call Gemini if download fails
        mock_client, _ = mock_genai_client
        assert not mock_client.models.generate_content.called

    def test_api_failure_fallback(self, mock_httpx_get, mock_genai_client):
        """Test that a Gemini API failure returns gracefully."""
        mock_client, _ = mock_genai_client
        mock_client.models.generate_content.side_effect = Exception("API down")
        
        result = analyze_image("http://fake.url/image.jpg")
        
        assert result.description == "Analysis failed due to an AI service error."
        assert result.confidence == 0.0
