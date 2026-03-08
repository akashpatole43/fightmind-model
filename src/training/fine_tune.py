"""
FightMind AI — Embedding Model Fine-Tuner
==========================================
Step 1.7 — Fine-tunes sentence-transformers/all-MiniLM-L6-v2 on domain-specific
martial arts text pairs to improve RAG retrieval accuracy.

Run on Google Colab (free T4 GPU) — see docs/COLAB_FINE_TUNE_GUIDE.md

Why fine-tune?
  The base model (all-MiniLM-L6-v2) was trained on general English text.
  By fine-tuning on martial arts Q&A pairs, embeddings of semantically
  similar martial arts concepts (e.g. "jab" ≈ "lead hand punch") move
  closer together in vector space, dramatically improving RAG precision.

Training Strategy:
  - Positive pairs  : two chunks from the SAME document (adjacent context)
  - In-batch negatives : MultipleNegativesRankingLoss treats all other
    batch items as implicit negatives — no explicit negative mining needed
  - Model output saved to models/fine_tuned/ and optionally to Google Drive

Usage:
    # Local (CPU only — slow, for testing):
    python -m src.training.fine_tune --epochs 1 --batch-size 8

    # Google Colab (see COLAB_FINE_TUNE_GUIDE.md):
    !python fine_tune.py --epochs 10 --batch-size 32 --output /content/drive/MyDrive/fightmind_model
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# Path breakdown from src/training/fine_tune.py:
#   parents[0] = src/training
#   parents[1] = src
#   parents[2] = fightmind-model  ← project root ✓
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT    = Path(__file__).resolve().parents[2]
PROCESSED_DIR    = _PROJECT_ROOT / "data" / "processed"
MODEL_OUTPUT_DIR = _PROJECT_ROOT / "models" / "fine_tuned"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Base Model
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Training Pair Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_training_pairs(
    chunks: list[dict],
    max_pairs: int = 10_000,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Build positive text pairs for MultipleNegativesRankingLoss.

    Positive pair strategy:
        Adjacent chunks from the SAME document share overlapping context
        (guaranteed by our 200-char overlap in preprocess.py), making them
        ideal semantic positives without any labeling effort.

    Args:
        chunks:    All chunks from data/processed/chunks.json
        max_pairs: Cap on total pairs (keeps Colab training time reasonable)
        seed:      Random seed for reproducible shuffling

    Returns:
        List of (anchor_text, positive_text) tuples.

    Raises:
        ValueError: if chunks list is empty
    """
    if not chunks:
        raise ValueError("chunks list is empty — run preprocess.py first")

    random.seed(seed)

    # Group chunks by doc_title so we can pair adjacent chunks from same doc
    from collections import defaultdict
    docs: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        key = f"{chunk.get('source', '')}::{chunk.get('doc_title', '')}"
        docs[key].append(chunk)

    pairs: list[tuple[str, str]] = []

    for doc_chunks in docs.values():
        # Sort by chunk_index to ensure adjacency
        doc_chunks.sort(key=lambda c: c.get("chunk_index", 0))

        # Pair each chunk with the next one in the same document
        for i in range(len(doc_chunks) - 1):
            anchor   = doc_chunks[i]["text"].strip()
            positive = doc_chunks[i + 1]["text"].strip()
            if anchor and positive:
                pairs.append((anchor, positive))

    logger.info(
        "Training pairs built",
        extra={"total_pairs": len(pairs), "unique_docs": len(docs)},
    )

    # Shuffle and cap
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]

    logger.info("Training pairs after cap", extra={"pairs": len(pairs), "max_pairs": max_pairs})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Fine-Tuning
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(
    chunks_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    base_model: str = BASE_MODEL_NAME,
    epochs: int = 10,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
    max_pairs: int = 10_000,
    seed: int = 42,
) -> Path:
    """
    Fine-tune the embedding model on martial arts text pairs.

    Steps:
        1. Load chunks from data/processed/chunks.json
        2. Build (anchor, positive) training pairs
        3. Load the base sentence-transformer model
        4. Train with MultipleNegativesRankingLoss
        5. Save the fine-tuned model to models/fine_tuned/

    Args:
        chunks_path: Path to chunks.json. Defaults to data/processed/chunks.json
        output_dir:  Where to save the model. Defaults to models/fine_tuned/
        base_model:  HuggingFace model name to fine-tune
        epochs:      Training epochs (10 recommended for Colab T4)
        batch_size:  Training batch size (32 fits in Colab T4 16GB VRAM)
        warmup_ratio: Fraction of steps used for LR warmup (stabilises early training)
        max_pairs:   Max number of training pairs to generate
        seed:        Random seed for reproducibility

    Returns:
        Path to the saved fine-tuned model directory.

    Raises:
        FileNotFoundError: if chunks.json not found
        ImportError:       if sentence-transformers not installed
    """
    # ── Lazy import: sentence-transformers is large, import only when training ──
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        logger.critical(
            "sentence-transformers not installed. Run: pip install sentence-transformers",
            exc_info=exc,
        )
        raise

    chunks_path = chunks_path or (PROCESSED_DIR / "chunks.json")
    output_dir  = output_dir  or MODEL_OUTPUT_DIR

    # ── Step 1: Load chunks ───────────────────────────────────────────────────
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {chunks_path}. "
            f"Run preprocess.py first: python -m src.training.preprocess"
        )

    logger.info("Loading chunks for fine-tuning", extra={"path": str(chunks_path)})
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info("Chunks loaded", extra={"count": len(chunks)})

    # ── Step 2: Build training pairs ──────────────────────────────────────────
    pairs = build_training_pairs(chunks, max_pairs=max_pairs, seed=seed)
    if not pairs:
        logger.error("No training pairs could be built from chunks")
        raise ValueError("No training pairs built — check that chunks.json has multi-chunk documents")

    # Convert to sentence-transformers InputExample format
    train_examples = [InputExample(texts=[a, p]) for a, p in pairs]

    # ── Step 3: Load base model ───────────────────────────────────────────────
    logger.info("Loading base model", extra={"model": base_model})
    model = SentenceTransformer(base_model)

    # ── Step 4: Configure training ────────────────────────────────────────────
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
    )

    # MultipleNegativesRankingLoss:
    #   - Treats all OTHER examples in the batch as negatives (in-batch negatives)
    #   - No need to manually create negative pairs
    #   - Works well for retrieval tasks where we want anchor ≈ positive
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)

    logger.info(
        "Starting fine-tuning",
        extra={
            "base_model": base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "training_pairs": len(train_examples),
            "output_dir": str(output_dir),
        },
    )

    # ── Step 5: Train and save ────────────────────────────────────────────────
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,     # visible in Colab
        checkpoint_path=str(output_dir / "checkpoints"),
        checkpoint_save_steps=len(train_dataloader),  # save after every epoch
    )

    logger.info(
        "Fine-tuning complete — model saved",
        extra={"output_dir": str(output_dir)},
    )
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune all-MiniLM-L6-v2 on FightMind martial arts chunks"
    )
    parser.add_argument("--chunks",     type=Path, default=None,       help="Path to chunks.json")
    parser.add_argument("--output",     type=Path, default=None,       help="Output directory for fine-tuned model")
    parser.add_argument("--model",      type=str,  default=BASE_MODEL_NAME, help="Base HuggingFace model name")
    parser.add_argument("--epochs",     type=int,  default=10,         help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int,  default=32,         help="Batch size (default: 32)")
    parser.add_argument("--max-pairs",  type=int,  default=10_000,     help="Max training pairs (default: 10000)")
    parser.add_argument("--seed",       type=int,  default=42,         help="Random seed (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    setup_logging()

    args = _parse_args()
    fine_tune(
        chunks_path=args.chunks,
        output_dir=args.output,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
