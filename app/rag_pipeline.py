"""
app/rag_pipeline.py

Day 1 core: turn raw HR/Labour-Law PDFs into a persisted ChromaDB
vector store.

Responsibilities:
    1. Discover and parse PDFs (PyMuPDF) into per-page text, preserving
       page numbers for later source citation.
    2. Chunk page text with LangChain's RecursiveCharacterTextSplitter,
       keeping page + source-file metadata attached to every chunk.
    3. Embed chunks with a local sentence-transformers model.
    4. Persist everything into a local ChromaDB collection.

This module has no LLM / chat dependency — it is purely the offline
ingestion pipeline, run once (or whenever the knowledge base changes)
via `ingest.py`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import os

import fitz  # PyMuPDF
from chromadb.config import Settings as ChromaClientSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)

# Disable ChromaDB's anonymized usage telemetry. It is harmless when it
# fails (a known posthog/chromadb version-mismatch issue — fixed for good
# by pinning posthog<3.0 in requirements.txt), but leaving telemetry off
# explicitly too avoids depending solely on that pin holding forever.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _build_chroma_client_settings() -> ChromaClientSettings:
    """
    Build a fresh ChromaDB client Settings object for a single Chroma store.

    IMPORTANT: this must return a NEW object on every call, never a shared
    module-level singleton. langchain_chroma mutates
    `client_settings.persist_directory` in place to match whatever
    `persist_directory` was passed to `Chroma(...)` — sharing one Settings
    instance across multiple stores (e.g. across tests using different
    temp directories) would let a later store's directory silently
    overwrite an earlier store's reference to the same object.

    `is_persistent=True` is set explicitly and is the critical part: when
    `client_settings` is provided to `Chroma(...)`, langchain_chroma does
    NOT default it to True the way it does when `persist_directory` is
    passed alone — omitting it silently produces an in-memory-only client
    that discards all data the moment the process exits, even though
    `persist_directory` is set and ingestion reports success.
    """
    return ChromaClientSettings(anonymized_telemetry=False, is_persistent=True)


@dataclass(frozen=True)
class PageRecord:
    """A single parsed PDF page with its extracted text and metadata."""

    source_file: str
    page_number: int  # 1-indexed, human-friendly
    text: str


class PDFParsingError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


class EmptyKnowledgeBaseError(RuntimeError):
    """Raised when no usable text was extracted from any document."""


def discover_pdfs(policies_dir: Optional[str] = None) -> List[Path]:
    """
    Find all PDF files under the policies directory.

    Args:
        policies_dir: Optional override of the directory to scan.
            Defaults to `settings.POLICIES_DIR`.

    Returns:
        Sorted list of PDF file paths (empty list if none found).
    """
    directory = Path(policies_dir or settings.POLICIES_DIR)
    if not directory.exists():
        logger.warning("Policies directory does not exist: %s", directory)
        return []

    pdf_paths = sorted(directory.glob("*.pdf"))
    logger.info("Discovered %d PDF(s) in %s", len(pdf_paths), directory)
    return pdf_paths


def extract_pages_from_pdf(pdf_path: Path) -> List[PageRecord]:
    """
    Extract per-page text from a single PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageRecord, one per non-empty page.

    Raises:
        PDFParsingError: If the file cannot be opened or read.
    """
    records: List[PageRecord] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                text = page.get_text("text").strip()
                if not text:
                    logger.debug(
                        "Skipping empty page %d in %s", page_index + 1, pdf_path.name
                    )
                    continue
                records.append(
                    PageRecord(
                        source_file=pdf_path.name,
                        page_number=page_index + 1,
                        text=text,
                    )
                )
    except Exception as exc:  # noqa: BLE001 - we deliberately wrap all fitz errors
        raise PDFParsingError(f"Failed to parse PDF '{pdf_path.name}': {exc}") from exc

    logger.info("Extracted %d non-empty page(s) from %s", len(records), pdf_path.name)
    return records


def build_documents(pages: List[PageRecord]) -> List[Document]:
    """
    Chunk page-level text into LangChain Documents, carrying forward
    source file + page number metadata plus a stable chunk id for
    idempotent re-ingestion.

    Args:
        pages: Flat list of parsed pages across all source PDFs.

    Returns:
        List of chunked LangChain Document objects ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: List[Document] = []
    for page in pages:
        chunks = splitter.split_text(page.text)
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = _make_chunk_id(page.source_file, page.page_number, chunk_index)
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source_file": page.source_file,
                        "page_number": page.page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                    },
                )
            )

    logger.info("Built %d chunk(s) from %d page(s)", len(documents), len(pages))
    return documents


def _make_chunk_id(source_file: str, page_number: int, chunk_index: int) -> str:
    """Deterministic id for a chunk, used as the ChromaDB document id."""
    raw = f"{source_file}::p{page_number}::c{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Build the sentence-transformers embedding function used for both
    ingestion and query-time retrieval. Centralized here so ingestion
    and retrieval never drift onto different embedding models.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_all(
    policies_dir: Optional[str] = None,
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    End-to-end ingestion: discover PDFs -> parse -> chunk -> embed -> persist.

    Idempotent: chunk ids are content-addressed (source file + page +
    chunk index), so re-running ingestion after adding/removing a PDF
    upserts rather than duplicating existing chunks.

    Args:
        policies_dir: Optional override of the PDF source directory.
        persist_directory: Optional override of the ChromaDB persist path
            (defaults to `settings.CHROMA_PERSIST_DIR`). Primarily for
            tests that need an isolated, disposable store.
        collection_name: Optional override of the ChromaDB collection name
            (defaults to `settings.CHROMA_COLLECTION_NAME`). Same rationale.

    Returns:
        The populated, persisted Chroma vector store instance.

    Raises:
        EmptyKnowledgeBaseError: If no PDFs were found or no text could
            be extracted from any of them.
    """
    pdf_paths = discover_pdfs(policies_dir)
    if not pdf_paths:
        raise EmptyKnowledgeBaseError(
            f"No PDF files found in '{policies_dir or settings.POLICIES_DIR}'. "
            "Add your HR/labour-law PDFs there before running ingestion."
        )

    all_pages: List[PageRecord] = []
    failed_files: List[str] = []
    for pdf_path in pdf_paths:
        try:
            all_pages.extend(extract_pages_from_pdf(pdf_path))
        except PDFParsingError as exc:
            logger.error(str(exc))
            failed_files.append(pdf_path.name)

    if failed_files:
        logger.warning(
            "%d file(s) failed to parse and were skipped: %s",
            len(failed_files),
            ", ".join(failed_files),
        )

    if not all_pages:
        raise EmptyKnowledgeBaseError(
            "No extractable text found in any PDF. Files may be scanned "
            "images without OCR text layers."
        )

    documents = build_documents(all_pages)
    if not documents:
        raise EmptyKnowledgeBaseError("Chunking produced zero documents; check chunk settings.")

    embedding_fn = get_embedding_function()
    chunk_ids = [doc.metadata["chunk_id"] for doc in documents]
    resolved_persist_dir = persist_directory or settings.CHROMA_PERSIST_DIR
    resolved_collection = collection_name or settings.CHROMA_COLLECTION_NAME

    logger.info(
        "Embedding and persisting %d chunk(s) into ChromaDB collection '%s' at '%s'",
        len(documents),
        resolved_collection,
        resolved_persist_dir,
    )
    vector_store = Chroma(
        collection_name=resolved_collection,
        embedding_function=embedding_fn,
        persist_directory=resolved_persist_dir,
        client_settings=_build_chroma_client_settings(),
        # Embeddings are L2-normalized (see get_embedding_function), so cosine
        # similarity is the correct space. This also makes Chroma's built-in
        # relevance-score conversion (used at query time for confidence
        # scoring) meaningful — without it, scores default to a raw L2
        # distance transform that does not sit cleanly in [0, 1].
        collection_metadata={"hnsw:space": "cosine"},
    )
    # `add_documents` with explicit ids upserts existing ids instead of
    # duplicating them, making re-ingestion safe to run repeatedly.
    vector_store.add_documents(documents=documents, ids=chunk_ids)

    logger.info("Ingestion complete. Collection now has %d chunk(s) total.", len(chunk_ids))
    return vector_store


def get_vector_store(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Chroma:
    """
    Load the already-persisted Chroma vector store for query-time use
    (used by app/vector_store.py at chat time — no re-ingestion here).

    Args:
        persist_directory: Optional override of the ChromaDB persist path
            (defaults to `settings.CHROMA_PERSIST_DIR`). Primarily for tests.
        collection_name: Optional override of the ChromaDB collection name
            (defaults to `settings.CHROMA_COLLECTION_NAME`). Same rationale.

    Returns:
        A Chroma instance backed by the on-disk persistent directory.
    """
    embedding_fn = get_embedding_function()
    return Chroma(
        collection_name=collection_name or settings.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=persist_directory or settings.CHROMA_PERSIST_DIR,
        client_settings=_build_chroma_client_settings(),
        collection_metadata={"hnsw:space": "cosine"},
    )