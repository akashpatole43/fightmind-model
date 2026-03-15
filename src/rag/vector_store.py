"""
FightMind AI — ChromaDB Vector Store
======================================
Step 1.8 — Loads data/processed/chunks.json, generates embeddings with the
fine-tuned sentence-transformer model, and persists everything to a ChromaDB
collection at embeddings/vectorstore/.

Public API
----------
    build(...)   — ingest / refresh the vector store
    search(...)  — semantic search with optional sport filter
    get_collection(...) — lightweight collection handle (no model load)

Example usage
-------------
    # Build once (or after data refresh)
    python -m src.rag.vector_store

    # Search from code
    from src.rag.vector_store import search
    results = search("how to throw a jab", sport="boxing", top_k=5)
    for r in results:
        print(r["score"], r["text"][:80])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths — resolved relative to the project root so the module works from any
# working directory (local dev, Docker, Colab).
#
# Path breakdown from src/rag/vector_store.py:
#   parents[0] = src/rag
#   parents[1] = src
#   parents[2] = fightmind-model  ← project root ✓
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH   = _PROJECT_ROOT / "data"       / "processed" / "chunks.json"
MODEL_DIR     = _PROJECT_ROOT / "models"     / "fine_tuned"
CHROMA_DIR    = _PROJECT_ROOT / "embeddings" / "vectorstore"

COLLECTION_NAME = "fightmind_chunks"
DEFAULT_BATCH   = 64          # chunks per ChromaDB upsert call
DEFAULT_TOP_K   = 5


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_dir: Path):
    """Load SentenceTransformer from *model_dir*.

    Falls back to the base HuggingFace model name if the local directory does
    not contain a fine-tuned model (useful for CI / first-run scenarios).
    """
    from sentence_transformers import SentenceTransformer  # lazy import

    if model_dir.exists() and (model_dir / "config.json").exists():
        logger.info("Loading fine-tuned embedding model", extra={"model_dir": str(model_dir)})
        return SentenceTransformer(str(model_dir))

    fallback = "sentence-transformers/all-MiniLM-L6-v2"
    logger.warning(
        "Fine-tuned model not found — falling back to base model",
        extra={"model_dir": str(model_dir), "fallback": fallback},
    )
    return SentenceTransformer(fallback)


def _get_client(persist_dir: Path):
    """Return a ChromaDB PersistentClient at *persist_dir*."""
    import chromadb  # lazy import

    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_collection(persist_dir: Path = CHROMA_DIR):
    """Return the ChromaDB collection without loading the embedding model.

    Useful for lightweight status checks from the API layer.

    Args:
        persist_dir: Path to the ChromaDB persistence directory.

    Returns:
        chromadb Collection object.

    Raises:
        FileNotFoundError: If *persist_dir* does not exist.
        Exception: If the collection has not been built yet.
    """
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. "
            "Run `python -m src.rag.vector_store` to build it first."
        )
    client = _get_client(persist_dir)
    return client.get_collection(name=COLLECTION_NAME)


def build(
    chunks_path: Path = CHUNKS_PATH,
    persist_dir: Path = CHROMA_DIR,
    model_dir:   Path = MODEL_DIR,
    batch_size:  int  = DEFAULT_BATCH,
    force_rebuild: bool = False,
) -> int:
    """Load chunks.json, embed every chunk, and upsert into ChromaDB.

    The collection is created the first time and reused on subsequent calls
    (idempotent upserts).  Pass ``force_rebuild=True`` to wipe and recreate
    the collection from scratch.

    Args:
        chunks_path:   Path to data/processed/chunks.json.
        persist_dir:   ChromaDB persistence directory.
        model_dir:     Directory of the fine-tuned SentenceTransformer model.
        batch_size:    Number of chunks to embed and upsert per batch.
        force_rebuild: Delete the collection before rebuilding if True.

    Returns:
        Total number of chunks upserted.

    Raises:
        FileNotFoundError: If chunks_path does not exist.
    """
    # ── Validate input ─────────────────────────────────────────────────────
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {chunks_path}. "
            "Run preprocess.py first: python -m src.training.preprocess"
        )

    # ── Load chunks ────────────────────────────────────────────────────────
    with open(chunks_path, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    if not chunks:
        logger.warning("chunks.json is empty — vector store will be empty")
        return 0

    logger.info("Chunks loaded", extra={"count": len(chunks), "path": str(chunks_path)})

    # ── ChromaDB client + collection ───────────────────────────────────────
    client = _get_client(persist_dir)

    if force_rebuild:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            logger.info("Existing collection deleted for rebuild", extra={"collection": COLLECTION_NAME})
        except Exception:
            pass  # collection didn't exist yet — that's fine

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for embeddings
    )

    # ── Skip if already populated ──────────────────────────────────────────
    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        logger.info(
            "Vector store already populated — skipping rebuild "
            "(use force_rebuild=True to refresh)",
            extra={"existing_chunks": existing_count, "collection": COLLECTION_NAME},
        )
        return existing_count

    # ── Load embedding model ────────────────────────────────────────────────
    model = _load_model(model_dir)

    # ── Embed and upsert in batches ─────────────────────────────────────────
    total_upserted = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        ids        = [c["chunk_id"] for c in batch]
        texts      = [c["text"]     for c in batch]
        metadatas  = [
            {
                "sport":       c.get("sport",       "general"),
                "source":      c.get("source",      "unknown"),
                "doc_title":   c.get("doc_title",   ""),
                "doc_url":     c.get("doc_url",      ""),
                "chunk_index": str(c.get("chunk_index", 0)),   # ChromaDB requires str/int/float/bool
            }
            for c in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_upserted += len(batch)
        logger.debug(
            "Batch upserted",
            extra={
                "batch_start": batch_start,
                "batch_size":  len(batch),
                "total_so_far": total_upserted,
            },
        )

    logger.info(
        "Vector store built",
        extra={
            "collection":     COLLECTION_NAME,
            "total_chunks":   total_upserted,
            "persist_dir":    str(persist_dir),
        },
    )
    return total_upserted


def search(
    query:       str,
    sport:       Optional[str] = None,
    top_k:       int           = DEFAULT_TOP_K,
    persist_dir: Path          = CHROMA_DIR,
    model_dir:   Path          = MODEL_DIR,
) -> list[dict]:
    """Semantic search over the vector store.

    Args:
        query:       Natural-language query string.
        sport:       Optional sport filter — one of "boxing", "kickboxing",
                     "karate", "general".  If None, searches all sports.
        top_k:       Maximum number of results to return.
        persist_dir: ChromaDB persistence directory.
        model_dir:   Directory of the fine-tuned SentenceTransformer model.

    Returns:
        List of result dicts, ordered by relevance (highest first)::

            [
                {
                    "chunk_id": "wikipedia_00042",
                    "text":     "A jab is a quick, straight punch ...",
                    "score":    0.91,        # cosine similarity (0–1)
                    "metadata": {
                        "sport":     "boxing",
                        "source":    "wikipedia",
                        "doc_title": "Jab (boxing)",
                        "doc_url":   "https://...",
                        "chunk_index": "0",
                    },
                },
                ...
            ]

    Raises:
        FileNotFoundError: If the vector store has not been built yet.
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to search — returning empty list")
        return []

    # ── Load model + embed query ───────────────────────────────────────────
    model          = _load_model(model_dir)
    query_embedding = model.encode([query.strip()], show_progress_bar=False).tolist()

    # ── Query ChromaDB ─────────────────────────────────────────────────────
    collection = get_collection(persist_dir)

    where = {"sport": sport} if sport else None

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed", exc_info=exc, extra={"query": query, "sport": sport})
        return []

    # ── Format results ─────────────────────────────────────────────────────
    output: list[dict] = []
    ids        = results.get("ids",        [[]])[0]
    docs       = results.get("documents",  [[]])[0]
    metadatas  = results.get("metadatas",  [[]])[0]
    distances  = results.get("distances",  [[]])[0]

    for chunk_id, text, meta, dist in zip(ids, docs, metadatas, distances):
        # ChromaDB cosine distance is in [0, 2]; convert to similarity [0, 1]
        score = max(0.0, 1.0 - dist / 2.0)
        output.append({
            "chunk_id": chunk_id,
            "text":     text,
            "score":    round(score, 4),
            "metadata": meta,
        })

    logger.info(
        "Search complete",
        extra={
            "query":   query[:60],
            "sport":   sport,
            "top_k":   top_k,
            "results": len(output),
        },
    )
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point — python -m src.rag.vector_store
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.core.logging_config import setup_logging

    setup_logging()

    force = "--force" in sys.argv
    if force:
        logger.info("Force-rebuild flag detected — existing collection will be deleted")

    count = build(force_rebuild=force)
    logger.info(
        "Vector store ready",
        extra={"total_chunks": count, "collection": COLLECTION_NAME},
    )
