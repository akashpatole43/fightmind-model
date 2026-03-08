"""
FightMind AI — Scraper Tests
==============================
Tests for src/data_collection/scraper.py.
All external HTTP calls are mocked — tests run offline, no API usage.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWikipediaScraper:

    @patch("src.data_collection.scraper.wikipedia.page")
    def test_scrape_returns_expected_fields(self, mock_page):
        """Each wikipedia record must have title, text, url, source."""
        from src.data_collection.scraper import scrape_wikipedia

        mock_page.return_value = MagicMock(
            title="Boxing",
            content="Boxing is a combat sport...",
            url="https://en.wikipedia.org/wiki/Boxing",
        )
        results = scrape_wikipedia(topics=["Boxing"])

        assert len(results) == 1
        assert results[0]["title"] == "Boxing"
        assert results[0]["source"] == "wikipedia"
        assert "text" in results[0]
        assert "url" in results[0]

    @patch("src.data_collection.scraper.wikipedia.page")
    def test_scrape_skips_on_page_error(self, mock_page):
        """PageError topics are gracefully skipped."""
        import wikipedia
        from src.data_collection.scraper import scrape_wikipedia

        mock_page.side_effect = wikipedia.PageError("NonExistent")
        results = scrape_wikipedia(topics=["NonExistentTopic123"])

        assert results == []

    @patch("src.data_collection.scraper.wikipedia.page")
    def test_scrape_multiple_topics(self, mock_page):
        """Multiple topics return multiple records."""
        from src.data_collection.scraper import scrape_wikipedia

        mock_page.return_value = MagicMock(
            title="Sport",
            content="Some sport content...",
            url="https://en.wikipedia.org/wiki/Sport",
        )
        results = scrape_wikipedia(topics=["Boxing", "Karate", "Kickboxing"])
        assert len(results) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Rules Pages Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRulesScraper:

    @patch("src.data_collection.scraper.SESSION")
    def test_scrape_rules_returns_text(self, mock_session):
        """Rules pages scraper must return title, text, url, source."""
        from src.data_collection.scraper import scrape_rules_pages

        # Mock HTML response with paragraph content
        html = "<html><body><article><p>Rule 1: Boxers must...</p><p>Rule 2: No biting.</p></article></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        pages = [{"name": "Test Rules", "url": "http://test.com/rules", "tag": "article"}]
        results = scrape_rules_pages(pages=pages)

        assert len(results) == 1
        assert results[0]["title"] == "Test Rules"
        assert results[0]["source"] == "official_rules"
        assert "Rule 1" in results[0]["text"]

    @patch("src.data_collection.scraper.SESSION")
    def test_scrape_rules_skips_on_http_error(self, mock_session):
        """HTTP errors are gracefully skipped."""
        import requests
        from src.data_collection.scraper import scrape_rules_pages

        mock_session.get.side_effect = requests.RequestException("Connection refused")
        pages = [{"name": "Bad URL", "url": "http://bad-url.test", "tag": "div"}]
        results = scrape_rules_pages(pages=pages)

        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# YouTube Transcript Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestYouTubeScraper:

    @patch("src.data_collection.scraper.YouTubeTranscriptApi.get_transcript")
    def test_scrape_transcript_returns_record(self, mock_transcript):
        """Transcript scraper must return cleaned text."""
        from src.data_collection.scraper import scrape_youtube_transcripts

        mock_transcript.return_value = [
            {"text": "Today we learn the jab.", "start": 0.0, "duration": 3.0},
            {"text": "Step forward with your lead foot.", "start": 3.0, "duration": 3.0},
        ]
        videos = [{"title": "Jab Tutorial", "video_id": "abc123"}]
        results = scrape_youtube_transcripts(videos=videos)

        assert len(results) == 1
        assert results[0]["title"] == "Jab Tutorial"
        assert results[0]["source"] == "youtube_transcript"
        assert "jab" in results[0]["text"].lower()

    @patch("src.data_collection.scraper.YouTubeTranscriptApi.get_transcript")
    def test_scrape_transcript_skips_disabled(self, mock_transcript):
        """Videos with disabled transcripts are gracefully skipped."""
        from youtube_transcript_api import TranscriptsDisabled
        from src.data_collection.scraper import scrape_youtube_transcripts

        mock_transcript.side_effect = TranscriptsDisabled("abc123")
        videos = [{"title": "No Transcript Video", "video_id": "abc123"}]
        results = scrape_youtube_transcripts(videos=videos)

        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Save Helper Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveHelpers:

    def test_save_json_creates_file(self, tmp_path, monkeypatch):
        """_save_json must write valid JSON to the target path."""
        from src.data_collection import scraper

        # Redirect RAW_DIR to pytest's tmp_path
        monkeypatch.setattr(scraper, "RAW_DIR", tmp_path)

        data = [{"title": "Test", "text": "content", "url": "http://x.com", "source": "test"}]
        path = scraper._save_json(data, "test_output.json")

        assert path.exists()
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded[0]["title"] == "Test"
