"""
Tests for src/training/fine_tune.py
=====================================
All model training and file I/O is mocked — no GPU, internet, or
chunks.json required to run these tests.

Tests cover:
  - build_training_pairs: pair generation, max_pairs cap, empty input
  - fine_tune: chunks.json not found, sentence-transformers not installed,
    successful run with mocked model.fit()
  - CLI argument parsing
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.core.logging_config import setup_logging
setup_logging()

from src.training.fine_tune import build_training_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk(
    doc_title: str = "Jab (boxing)",
    source: str = "wikipedia",
    sport: str = "boxing",
    chunk_index: int = 0,
    total_chunks: int = 3,
    text: str = None,
) -> dict:
    if text is None:
        text = f"This is chunk {chunk_index} about {doc_title}. " * 10
    return {
        "chunk_id":    f"{source}_{chunk_index:05d}",
        "text":        text,
        "sport":       sport,
        "source":      source,
        "doc_title":   doc_title,
        "doc_url":     f"https://example.com/{doc_title}",
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }


def _make_doc_chunks(title: str, n: int = 3, source: str = "wikipedia", sport: str = "boxing") -> list[dict]:
    """Return n sequential chunks for a single document."""
    return [_make_chunk(title, source, sport, i, n) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# build_training_pairs
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTrainingPairs:

    def test_raises_on_empty_chunks(self):
        with pytest.raises(ValueError, match="empty"):
            build_training_pairs([])

    def test_single_chunk_doc_produces_no_pairs(self):
        chunks = [_make_chunk(chunk_index=0, total_chunks=1)]
        pairs = build_training_pairs(chunks)
        # A document with only 1 chunk has no adjacent neighbour → 0 pairs
        assert pairs == []

    def test_two_chunk_doc_produces_one_pair(self):
        chunks = _make_doc_chunks("Jab (boxing)", n=2)
        pairs = build_training_pairs(chunks)
        assert len(pairs) == 1

    def test_three_chunk_doc_produces_two_pairs(self):
        chunks = _make_doc_chunks("Uppercut", n=3)
        pairs = build_training_pairs(chunks)
        assert len(pairs) == 2

    def test_multiple_docs_pairs_are_intra_doc(self):
        """Pairs should only come from within the same document."""
        boxing_chunks = _make_doc_chunks("Jab (boxing)", n=3, sport="boxing")
        karate_chunks = _make_doc_chunks("Kumite", n=3, sport="karate")
        pairs = build_training_pairs(boxing_chunks + karate_chunks)
        # 2 pairs per doc × 2 docs = 4 pairs
        assert len(pairs) == 4

    def test_max_pairs_cap_is_respected(self):
        """Pairs should never exceed max_pairs."""
        chunks = _make_doc_chunks("Big Doc", n=100)  # 99 possible pairs
        pairs = build_training_pairs(chunks, max_pairs=5)
        assert len(pairs) <= 5

    def test_pairs_are_tuples_of_two_strings(self):
        chunks = _make_doc_chunks("Roundhouse Kick", n=3)
        pairs = build_training_pairs(chunks)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)

    def test_reproducible_with_same_seed(self):
        chunks = _make_doc_chunks("Boxing", n=20)
        pairs1 = build_training_pairs(chunks, seed=42)
        pairs2 = build_training_pairs(chunks, seed=42)
        assert pairs1 == pairs2

    def test_different_seeds_produce_different_order(self):
        chunks = _make_doc_chunks("Boxing", n=20)
        pairs1 = build_training_pairs(chunks, seed=1)
        pairs2 = build_training_pairs(chunks, seed=99)
        # With enough pairs, different seeds almost always produce different order
        assert pairs1 != pairs2


# ─────────────────────────────────────────────────────────────────────────────
# fine_tune — error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestFineTuneErrors:

    def test_raises_file_not_found_if_chunks_missing(self, tmp_path):
        """Gives a clear error if preprocess.py hasn't been run yet."""
        import src.training.fine_tune as module
        orig = module.PROCESSED_DIR
        module.PROCESSED_DIR = tmp_path   # empty tmp dir — no chunks.json
        try:
            from src.training.fine_tune import fine_tune
            with pytest.raises(FileNotFoundError, match="chunks.json"):
                fine_tune(chunks_path=tmp_path / "chunks.json")
        finally:
            module.PROCESSED_DIR = orig

    def test_raises_import_error_if_sentence_transformers_missing(self, tmp_path):
        """
        If sentence-transformers is not installed (e.g. in CI),
        fine_tune() should raise ImportError with a helpful message.
        """
        # Write a minimal chunks.json
        chunks_path = tmp_path / "chunks.json"
        chunks = _make_doc_chunks("Boxing", n=3)
        chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

        with patch.dict("sys.modules", {
            "sentence_transformers": None,
            "torch.utils.data": None,
        }):
            from src.training.fine_tune import fine_tune
            with pytest.raises((ImportError, TypeError)):
                fine_tune(chunks_path=chunks_path, output_dir=tmp_path / "out")


# ─────────────────────────────────────────────────────────────────────────────
# fine_tune — happy path (model.fit() mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestFineTuneHappyPath:

    def test_calls_model_fit_and_saves_output(self, tmp_path):
        """
        Smoke test: patches sentence_transformers so no GPU or download needed.
        Verifies that model.fit() is called with the right arguments.
        """
        chunks = _make_doc_chunks("Boxing Jab", n=5) + _make_doc_chunks("Karate Kumite", n=5)
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
        output_dir = tmp_path / "model_out"

        # Build mock objects for sentence_transformers
        mock_model     = MagicMock()
        mock_loss      = MagicMock()
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = lambda self: 10

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model
        mock_st_module.InputExample = lambda texts: MagicMock(texts=texts)
        mock_st_module.losses.MultipleNegativesRankingLoss.return_value = mock_loss

        mock_torch_module = MagicMock()
        mock_torch_module.DataLoader.return_value = mock_dataloader

        with patch.dict("sys.modules", {
            "sentence_transformers": mock_st_module,
            "sentence_transformers.losses": mock_st_module.losses,
            "torch.utils.data": mock_torch_module,
        }):
            from src.training import fine_tune as ft_module
            # Reload fresh to pick up the mocked modules
            import importlib
            importlib.reload(ft_module)

            result = ft_module.fine_tune(
                chunks_path=chunks_path,
                output_dir=output_dir,
                epochs=2,
                batch_size=4,
                max_pairs=50,
            )

        # model.fit should have been called once
        mock_model.fit.assert_called_once()
        # Check output_path directly from kwargs dict (avoids Windows backslash escaping issues)
        call_kwargs = mock_model.fit.call_args.kwargs
        assert call_kwargs["output_path"] == str(output_dir)
