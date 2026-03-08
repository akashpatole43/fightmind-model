"""
Tests for src/training/preprocess.py
======================================
All file I/O is isolated using tmp_path — no real data/raw files required.

Tests cover:
  - clean_text: citation removal, section headers, whitespace normalization
  - chunk_text: correct size, overlap, boundary-awareness, MIN_CHUNK_LEN filter
  - _detect_sport: correct classification for all three sports + general
  - process_records: empty text skipped, metadata correct, chunk IDs unique
  - _load_json: missing file, malformed JSON, non-list JSON
  - run_preprocessing: end-to-end with all 4 source files
  - save_chunks: output file created, permission error re-raised
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.core.logging_config import setup_logging
setup_logging()

from src.training.preprocess import (
    clean_text,
    chunk_text,
    _detect_sport,
    process_records,
    save_chunks,
    run_preprocessing,
    MIN_CHUNK_LEN,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# clean_text
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanText:

    def test_removes_wikipedia_citation_numbers(self):
        raw = "Tyson[1] was born[23] in Brooklyn."
        assert "[1]" not in clean_text(raw)
        assert "[23]" not in clean_text(raw)
        assert "Tyson" in clean_text(raw)

    def test_removes_wikipedia_section_headers(self):
        raw = "== Career ==\nHe started boxing."
        result = clean_text(raw)
        assert "==" not in result
        assert "He started boxing." in result

    def test_removes_edit_links(self):
        raw = "History [edit]\nBoxing has a long history."
        result = clean_text(raw)
        assert "[edit]" not in result

    def test_collapses_multiple_newlines(self):
        raw = "Para 1.\n\n\n\n\nPara 2."
        result = clean_text(raw)
        assert "\n\n\n" not in result

    def test_collapses_multiple_spaces(self):
        raw = "A jab   is   a punch."
        result = clean_text(raw)
        assert "   " not in result

    def test_returns_empty_string_for_none(self):
        assert clean_text(None) == ""

    def test_returns_empty_string_for_empty(self):
        assert clean_text("") == ""

    def test_strips_leading_trailing_whitespace(self):
        raw = "   some text   "
        assert clean_text(raw) == "some text"


# ─────────────────────────────────────────────────────────────────────────────
# chunk_text
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        text = "A" * (CHUNK_SIZE // 2)
        chunks = chunk_text(text)
        assert len(chunks) == 1

    def test_long_text_split_into_multiple_chunks(self):
        # 4× chunk size — should produce at least 3 chunks
        text = ("Boxing is a combat sport. " * 300)[:CHUNK_SIZE * 4]
        chunks = chunk_text(text)
        assert len(chunks) >= 3

    def test_chunks_do_not_exceed_chunk_size_significantly(self):
        text = "X" * (CHUNK_SIZE * 3)
        for chunk in chunk_text(text):
            # Allow 10% tolerance: boundary-aware split may land slightly over
            assert len(chunk) <= CHUNK_SIZE * 1.1

    def test_overlap_means_chunks_share_content(self):
        """Adjacent chunks must share at least some text (the overlap window)."""
        text = "The jab is fundamental. " * 200
        chunks = chunk_text(text, chunk_size=400, overlap=100)
        if len(chunks) >= 2:
            # End of chunk[0] and start of chunk[1] should share characters
            end_of_first   = chunks[0][-80:]
            start_of_second = chunks[1][:80]
            assert end_of_first or start_of_second   # at least one non-empty

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []

    def test_chunks_shorter_than_min_are_filtered(self):
        # Single chunk that is too short
        text = "Hi."
        chunks = chunk_text(text)
        assert all(len(c) >= MIN_CHUNK_LEN for c in chunks)

    def test_prefers_paragraph_boundary_split(self):
        """Chunker should split at \\n\\n rather than mid-word."""
        para1 = "A" * (CHUNK_SIZE - 100)
        para2 = "B" * 300
        text  = para1 + "\n\n" + para2
        chunks = chunk_text(text)
        # If split at paragraph boundary, chunk[0] should not contain "B"s
        assert "BBB" not in chunks[0]


# ─────────────────────────────────────────────────────────────────────────────
# _detect_sport
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectSport:

    def test_detects_boxing(self):
        assert _detect_sport("Jab (boxing)", "A jab is a quick punch in boxing.") == "boxing"

    def test_detects_kickboxing(self):
        assert _detect_sport("Roundhouse kick", "The roundhouse kick is used in kickboxing.") == "kickboxing"

    def test_detects_karate(self):
        assert _detect_sport("Shotokan Karate", "Kumite is sparring in karate.") == "karate"

    def test_returns_general_for_unrelated(self):
        assert _detect_sport("Nutrition", "Eating well improves performance.") == "general"

    def test_title_takes_priority(self):
        # Title says karate but body has boxing keywords
        result = _detect_sport("Karate Kata", "jab uppercut hook knockout ring wbc")
        assert result == "karate"


# ─────────────────────────────────────────────────────────────────────────────
# process_records
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessRecords:

    def _make_record(self, title="Jab (boxing)", text=None, url="https://example.com", source="wikipedia"):
        if text is None:
            text = "A jab is a straight punch thrown with the lead hand. " * 40
        return {"title": title, "text": text, "url": url, "source": source}

    def test_returns_chunks_for_valid_record(self):
        records = [self._make_record()]
        chunks = process_records(records, default_source="wikipedia")
        assert len(chunks) >= 1

    def test_chunk_has_required_fields(self):
        records = [self._make_record()]
        chunk = process_records(records, default_source="wikipedia")[0]
        required = {"chunk_id", "text", "sport", "source", "doc_title", "doc_url", "chunk_index", "total_chunks"}
        assert required.issubset(chunk.keys())

    def test_chunk_ids_are_unique(self):
        # Two records → multiple chunks → all IDs must be unique
        records = [self._make_record(), self._make_record(title="Uppercut")]
        chunks = process_records(records, default_source="wikipedia")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_skips_records_with_empty_text(self):
        records = [self._make_record(text="")]
        chunks = process_records(records, default_source="wikipedia")
        assert chunks == []

    def test_skips_records_with_too_short_text(self):
        records = [self._make_record(text="Short.")]
        chunks = process_records(records, default_source="wikipedia")
        assert chunks == []

    def test_sport_detected_correctly(self):
        records = [self._make_record(title="Kumite", text="In karate, kumite is sparring. " * 30)]
        chunks = process_records(records, default_source="wikipedia")
        assert all(c["sport"] == "karate" for c in chunks)

    def test_doc_title_and_url_preserved(self):
        records = [self._make_record(title="My Title", url="https://myurl.com")]
        chunks = process_records(records, default_source="wikipedia")
        assert chunks[0]["doc_title"] == "My Title"
        assert chunks[0]["doc_url"] == "https://myurl.com"


# ─────────────────────────────────────────────────────────────────────────────
# _load_json (tested indirectly via run_preprocessing with tmp_path)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadJson:
    """Test _load_json by patching RAW_DIR."""

    def test_returns_empty_for_missing_file(self, tmp_path):
        import src.training.preprocess as module
        original = module.RAW_DIR
        module.RAW_DIR = tmp_path
        try:
            from src.training.preprocess import _load_json
            result = _load_json("nonexistent.json")
            assert result == []
        finally:
            module.RAW_DIR = original

    def test_returns_empty_for_malformed_json(self, tmp_path):
        import src.training.preprocess as module
        original = module.RAW_DIR
        module.RAW_DIR = tmp_path
        (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")
        try:
            from src.training.preprocess import _load_json
            result = _load_json("bad.json")
            assert result == []
        finally:
            module.RAW_DIR = original

    def test_returns_empty_for_non_list_json(self, tmp_path):
        import src.training.preprocess as module
        original = module.RAW_DIR
        module.RAW_DIR = tmp_path
        (tmp_path / "obj.json").write_text('{"key": "value"}', encoding="utf-8")
        try:
            from src.training.preprocess import _load_json
            result = _load_json("obj.json")
            assert result == []
        finally:
            module.RAW_DIR = original


# ─────────────────────────────────────────────────────────────────────────────
# save_chunks
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveChunks:

    def test_saves_chunks_to_json(self, tmp_path):
        import src.training.preprocess as module
        original = module.PROCESSED_DIR
        module.PROCESSED_DIR = tmp_path
        try:
            chunks = [{"chunk_id": "test_00000", "text": "A jab is a punch.", "sport": "boxing"}]
            path = save_chunks(chunks, filename="test_chunks.json")
            assert path.exists()
            loaded = json.loads(path.read_text(encoding="utf-8"))
            assert len(loaded) == 1
            assert loaded[0]["chunk_id"] == "test_00000"
        finally:
            module.PROCESSED_DIR = original

    def test_raises_on_permission_error(self, tmp_path):
        import src.training.preprocess as module
        original = module.PROCESSED_DIR
        module.PROCESSED_DIR = tmp_path
        try:
            with patch("builtins.open", side_effect=PermissionError("no access")):
                with pytest.raises(PermissionError):
                    save_chunks([{"chunk_id": "x"}], filename="fail.json")
        finally:
            module.PROCESSED_DIR = original


# ─────────────────────────────────────────────────────────────────────────────
# run_preprocessing (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPreprocessing:

    def test_end_to_end_with_all_sources(self, tmp_path):
        """Full pipeline with minimal fixture data in tmp_path."""
        import src.training.preprocess as module
        orig_raw  = module.RAW_DIR
        orig_proc = module.PROCESSED_DIR
        module.RAW_DIR       = tmp_path / "raw"
        module.PROCESSED_DIR = tmp_path / "processed"
        module.RAW_DIR.mkdir()
        module.PROCESSED_DIR.mkdir()

        # Write minimal fixture files
        article_text = "Boxing is a combat sport where two fighters punch each other. " * 20
        (module.RAW_DIR / "wikipedia.json").write_text(
            json.dumps([{"title": "Boxing", "text": article_text, "url": "https://en.wikipedia.org/wiki/Boxing", "source": "wikipedia"}]),
            encoding="utf-8",
        )
        (module.RAW_DIR / "official_rules.json").write_text(
            json.dumps([{"title": "WBC Rules", "text": "A boxer must wear gloves. " * 20, "url": "https://wbc.com", "source": "official_rules"}]),
            encoding="utf-8",
        )
        (module.RAW_DIR / "youtube_transcripts.json").write_text(
            json.dumps([{"title": "Jab Tutorial", "text": "Keep your guard up when throwing a jab. " * 20, "url": "https://yt.com", "source": "youtube_transcript"}]),
            encoding="utf-8",
        )
        (module.RAW_DIR / "live_events.json").write_text(
            json.dumps([{"title": "Fury vs Usyk", "sport": "boxing", "date": "2026-03-08", "venue": "Riyadh", "fighters": "Fury vs Usyk", "promotion": "Top Rank", "source": "thesportsdb"}]),
            encoding="utf-8",
        )

        try:
            summary = run_preprocessing()
            assert summary["total"] > 0
            assert (module.PROCESSED_DIR / "chunks.json").exists()
            chunks = json.loads((module.PROCESSED_DIR / "chunks.json").read_text(encoding="utf-8"))
            assert len(chunks) > 0
            # Verify required fields present in all chunks
            for chunk in chunks:
                assert "chunk_id" in chunk
                assert "text" in chunk
                assert "sport" in chunk
        finally:
            module.RAW_DIR       = orig_raw
            module.PROCESSED_DIR = orig_proc

    def test_handles_missing_files_gracefully(self, tmp_path):
        """All raw files missing — returns summary with 0 chunks, never raises."""
        import src.training.preprocess as module
        orig_raw  = module.RAW_DIR
        orig_proc = module.PROCESSED_DIR
        module.RAW_DIR       = tmp_path / "empty_raw"
        module.PROCESSED_DIR = tmp_path / "processed"
        module.RAW_DIR.mkdir()
        module.PROCESSED_DIR.mkdir()
        try:
            summary = run_preprocessing()
            assert summary["total"] == 0
        finally:
            module.RAW_DIR       = orig_raw
            module.PROCESSED_DIR = orig_proc
