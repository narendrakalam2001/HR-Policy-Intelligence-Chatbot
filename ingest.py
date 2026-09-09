"""
ingest.py

Standalone runner script: parses every PDF in `data/policies/`, chunks
and embeds them, and persists the result into the local ChromaDB store.

Run this once before starting the chatbot, and again any time you add,
remove, or update a policy PDF.

Usage:
    python ingest.py
    python ingest.py --policies-dir data/policies --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.rag_pipeline import EmptyKnowledgeBaseError, ingest_all
from config import settings

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ingestion run."""
    parser = argparse.ArgumentParser(
        description="Ingest HR policy / labour law PDFs into ChromaDB."
    )
    parser.add_argument(
        "--policies-dir",
        type=str,
        default=None,
        help=f"Directory containing PDF files (default: {settings.POLICIES_DIR})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for this run.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Entry point. Returns a process exit code (0 = success, 1 = failure)
    so this script behaves correctly in CI pipelines / shell scripts.
    """
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting ingestion run...")
    logger.info("Policies directory : %s", args.policies_dir or settings.POLICIES_DIR)
    logger.info("Chroma persist dir : %s", settings.CHROMA_PERSIST_DIR)
    logger.info("Embedding model    : %s", settings.EMBEDDING_MODEL_NAME)
    logger.info("Chunk size/overlap : %d / %d", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    try:
        vector_store = ingest_all(policies_dir=args.policies_dir)
    except EmptyKnowledgeBaseError as exc:
        logger.error("Ingestion aborted: %s", exc)
        return 1
    except Exception:  # noqa: BLE001 - top-level guard so failures are logged, not a raw traceback
        logger.exception("Unexpected error during ingestion.")
        return 1

    try:
        count = vector_store._collection.count()  # noqa: SLF001 - quick sanity readout only
        logger.info("✅ Ingestion succeeded. Collection contains %d chunk(s).", count)
    except Exception:  # noqa: BLE001 - count() is a best-effort sanity check, not critical path
        logger.info("✅ Ingestion succeeded.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
