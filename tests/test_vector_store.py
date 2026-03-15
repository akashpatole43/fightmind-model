"""
Offline tests for the ChromaDB Vector Store (Step 1.8).

Uses unittest.mock to mock:
1. chromadb.PersistentClient (no disk I/O)
2. SentenceTransformer (no download or GPU usage)

This ensures CI is fast and deterministic.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.vector_store import build, get_collection, search


@pytest.fixture
def mock_chroma():
    """Mocks chromadb.PersistentClient and its collections."""
    with patch("chromadb.PersistentClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = mock_collection
        
        yield mock_client_cls, mock_client, mock_collection


@pytest.fixture
def mock_sentence_transformer():
    """Mocks SentenceTransformer to return dummy embeddings."""
    with patch("src.rag.vector_store._load_model") as mock_load:
        mock_model = MagicMock()
        # Return an array matching the input size, simulating 384-dimensional embeddings (reduced for test)
        mock_model.encode.side_effect = lambda texts, **_: type('MockArray', (), {'tolist': lambda: [[0.1, 0.2, 0.3]] * len(texts)})
        mock_load.return_value = mock_model
        yield mock_load, mock_model


@pytest.fixture
def dummy_chunks_file(tmp_path: Path):
    """Creates a temporary chunks.json file for testing."""
    chunks = [
        {
            "chunk_id": "test_001",
            "text": "A jab is a quick punch.",
            "sport": "boxing",
            "source": "wikipedia",
            "doc_title": "Jab",
            "doc_url": "",
            "chunk_index": 0,
        },
        {
            "chunk_id": "test_002",
            "text": "A roundhouse kick is powerful.",
            "sport": "kickboxing",
            "source": "wikipedia",
            "doc_title": "Roundhouse",
            "doc_url": "",
            "chunk_index": 0,
        }
    ]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks), encoding="utf-8")
    return path


class TestBuildVectorStore:
    def test_build_happy_path(self, mock_chroma, mock_sentence_transformer, dummy_chunks_file, tmp_path):
        """Tests that chunks are loaded, embedded, and upserted in batches."""
        _, _, mock_collection = mock_chroma
        mock_collection.count.return_value = 0  # Simulates empty collection

        persist_dir = tmp_path / "chroma"
        model_dir = tmp_path / "model"
        
        count = build(
            chunks_path=dummy_chunks_file,
            persist_dir=persist_dir,
            model_dir=model_dir,
            batch_size=2  # Process all at once
        )

        assert count == 2
        mock_collection.upsert.assert_called_once()
        kwargs = mock_collection.upsert.call_args.kwargs
        
        assert kwargs["ids"] == ["test_001", "test_002"]
        assert kwargs["documents"] == ["A jab is a quick punch.", "A roundhouse kick is powerful."]
        assert len(kwargs["embeddings"]) == 2
        
        assert kwargs["metadatas"][0]["sport"] == "boxing"
        assert kwargs["metadatas"][1]["sport"] == "kickboxing"

    def test_build_skips_if_already_populated(self, mock_chroma, mock_sentence_transformer, dummy_chunks_file, tmp_path):
        """Tests that build aborts natively if the collection has items, unless forced."""
        _, _, mock_collection = mock_chroma
        mock_collection.count.return_value = 100  # Already has items

        count = build(chunks_path=dummy_chunks_file, persist_dir=tmp_path, model_dir=tmp_path)

        assert count == 100
        mock_collection.upsert.assert_not_called()

    def test_build_force_rebuild(self, mock_chroma, mock_sentence_transformer, dummy_chunks_file, tmp_path):
        """Tests that force=True deletes existing collection and rebuilds."""
        _, mock_client, mock_collection = mock_chroma
        mock_collection.count.return_value = 100  # Will be bypassed

        count = build(
            chunks_path=dummy_chunks_file, 
            persist_dir=tmp_path, 
            model_dir=tmp_path, 
            force_rebuild=True
        )

        assert count == 2
        mock_client.delete_collection.assert_called_once_with(name="fightmind_chunks")
        mock_collection.upsert.assert_called_once()

    def test_build_missing_chunks_file(self, tmp_path):
        """Tests that FileNotFoundError is raised if chunks.json is missing."""
        with pytest.raises(FileNotFoundError):
            build(chunks_path=tmp_path / "does_not_exist.json", persist_dir=tmp_path, model_dir=tmp_path)


class TestSearchVectorStore:
    def test_search_happy_path(self, mock_chroma, mock_sentence_transformer, tmp_path):
        """Tests standard search flow returning formatted dicts."""
        _, _, mock_collection = mock_chroma
        
        # Mock ChromaDB query response
        mock_collection.query.return_value = {
            "ids": [["test_001"]],
            "documents": [["A jab is a quick punch."]],
            "metadatas": [[{"sport": "boxing", "doc_title": "Jab"}]],
            "distances": [[0.5]]  # Chroma returns distance, score = max(0, 1 - dist/2) -> 1 - 0.25 = 0.75
        }

        results = search(
            query="how to throw a jab",
            sport="boxing",
            top_k=1,
            persist_dir=tmp_path,
            model_dir=tmp_path
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "test_001"
        assert results[0]["text"] == "A jab is a quick punch."
        assert results[0]["score"] == 0.75
        assert results[0]["metadata"]["sport"] == "boxing"

        # Verify ChromaDB get queried correctly with `where` filter
        mock_collection.query.assert_called_once()
        kwargs = mock_collection.query.call_args.kwargs
        assert kwargs["n_results"] == 1
        assert kwargs["where"] == {"sport": "boxing"}
        assert len(kwargs["query_embeddings"]) == 1

    def test_search_empty_query_returns_empty(self, mock_chroma, mock_sentence_transformer, tmp_path):
        """Tests that empty or blank queries return empty lists correctly without erroring."""
        assert search("", persist_dir=tmp_path, model_dir=tmp_path) == []
        assert search("   \n ", persist_dir=tmp_path, model_dir=tmp_path) == []

    def test_search_no_sport_filter(self, mock_chroma, mock_sentence_transformer, tmp_path):
        """Tests that sport=None passes where=None to ChromaDB."""
        _, _, mock_collection = mock_chroma
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        search("punch", sport=None, persist_dir=tmp_path, model_dir=tmp_path)
        
        kwargs = mock_collection.query.call_args.kwargs
        assert kwargs["where"] is None


class TestGetCollection:
    def test_get_collection_missing_dir(self, tmp_path):
        """Tests that FileNotFoundError is raised if persist_dir does not exist."""
        with pytest.raises(FileNotFoundError):
            get_collection(persist_dir=tmp_path / "not_there")

    def test_get_collection_returns_collection(self, mock_chroma, tmp_path):
        """Tests that the collection object is returned successfully."""
        persist_dir = tmp_path / "chroma"
        persist_dir.mkdir()
        
        _, _, mock_collection = mock_chroma
        
        collection = get_collection(persist_dir)
        assert collection is mock_collection
