"""
app/vector_store.py

Query-time retrieval layer: wraps the persisted Chroma store from
app/rag_pipeline.py with:
    - lightweight query preprocessing (normalization, HR-domain
      abbreviation expansion, fuzzy spelling correction)
    - top-k retrieval with normalized relevance scores
    - a simple out-of-scope guardrail based on the top relevance score

Kept separate from app/llm_chain.py so retrieval logic can be unit
tested and reused without spinning up the LLM.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag_pipeline import get_vector_store
from config import settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Query preprocessing
# --------------------------------------------------------------------------- #

# Common HR/labour-law abbreviations expanded so the embedding model sees
# the full phrase — small local dictionaries like this outperform a full
# LLM rewrite call for a closed, well-known domain vocabulary, and add no
# extra API latency/cost.
ABBREVIATION_EXPANSIONS = {
    r"\bposh\b": "POSH (Prevention of Sexual Harassment)",
    r"\bpf\b": "PF (Provident Fund)",
    r"\bepf\b": "EPF (Employee Provident Fund)",
    r"\besi\b": "ESI (Employee State Insurance)",
    r"\bml\b": "maternity leave",
    r"\bpl\b": "privilege leave",
    r"\bcl\b": "casual leave",
    r"\bsl\b": "sick leave",
    r"\bwfh\b": "work from home",
    r"\bctc\b": "CTC (Cost to Company)",
    r"\bnotice period\b": "notice period",
}

# Domain vocabulary used for fuzzy spelling correction. Kept intentionally
# small and specific to this knowledge base rather than a general English
# dictionary — a general spellchecker would "correct" domain terms like
# "gratuity" or "POSH" into unrelated words.
DOMAIN_VOCABULARY = [
    "maternity", "benefit", "gratuity", "leave", "policy", "harassment",
    "posh", "labour", "law", "establishment", "shops", "wages", "salary",
    "notice", "period", "resignation", "termination", "compliance",
    "provident", "fund", "insurance", "bonus", "overtime", "holiday",
    "workplace", "committee", "complaint", "employee", "employer",
]


def _expand_abbreviations(text: str) -> str:
    """Expand known HR abbreviations (case-insensitive, whole-word only)."""
    expanded = text
    for pattern, replacement in ABBREVIATION_EXPANSIONS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def _correct_spelling(text: str) -> str:
    """
    Fuzzy-correct individual words against the domain vocabulary.

    Only replaces a word when a close match exists (cutoff=0.8) and the
    word isn't already a vocabulary hit — conservative on purpose, since
    an over-eager correction can silently change the user's intent.
    """
    words = text.split()
    corrected_words = []
    for word in words:
        bare = re.sub(r"[^\w]", "", word).lower()
        if not bare or bare in DOMAIN_VOCABULARY:
            corrected_words.append(word)
            continue
        matches = difflib.get_close_matches(bare, DOMAIN_VOCABULARY, n=1, cutoff=0.8)
        if matches:
            logger.debug("Query spelling correction: '%s' -> '%s'", bare, matches[0])
            corrected_words.append(matches[0])
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)


def preprocess_query(raw_query: str) -> str:
    """
    Normalize, spell-correct, and expand a raw user query before embedding.

    Args:
        raw_query: The exact text the user typed.

    Returns:
        A cleaned query string optimized for retrieval. Falls back to the
        (whitespace-normalized) original text if preprocessing empties it.
    """
    normalized = re.sub(r"\s+", " ", raw_query).strip()
    if not normalized:
        return normalized

    expanded = _expand_abbreviations(normalized)
    corrected = _correct_spelling(expanded)
    result = corrected.strip() or normalized

    if result != normalized:
        logger.info("Preprocessed query: %r -> %r", normalized, result)
    return result


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk paired with its normalized relevance score."""

    text: str
    source_file: str
    page_number: int
    relevance_score: float  # 0.0 (irrelevant) - 1.0 (highly relevant)


class Retriever:
    """
    Thin, testable wrapper around the persisted Chroma vector store for
    query-time retrieval with relevance scoring and an out-of-scope check.
    """

    def __init__(self, vector_store: Chroma | None = None) -> None:
        """
        Args:
            vector_store: Optional pre-built Chroma instance (useful for
                tests, e.g. injecting a store with a fake embedding
                function). Defaults to the persisted production store.
        """
        self._vector_store = vector_store or get_vector_store()

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a (preprocessed) query.

        Args:
            query: The query text (already preprocessed by the caller).
            top_k: Number of chunks to retrieve. Defaults to
                `settings.RETRIEVAL_TOP_K`.

        Returns:
            List of RetrievedChunk, most relevant first. Empty list if the
            store has no documents yet.
        """
        k = top_k or settings.RETRIEVAL_TOP_K
        try:
            results: List[Tuple[Document, float]] = (
                self._vector_store.similarity_search_with_relevance_scores(query, k=k)
            )
        except Exception:
            logger.exception("Retrieval failed for query: %r", query)
            return []

        chunks = [
            RetrievedChunk(
                text=doc.page_content,
                source_file=doc.metadata.get("source_file", "unknown"),
                page_number=doc.metadata.get("page_number", -1),
                relevance_score=max(0.0, min(1.0, float(score))),
            )
            for doc, score in results
        ]
        logger.info(
            "Retrieved %d chunk(s) for query %r (top score: %.3f)",
            len(chunks),
            query,
            chunks[0].relevance_score if chunks else 0.0,
        )
        return chunks

    def is_in_scope(self, chunks: List[RetrievedChunk]) -> bool:
        """
        Guardrail: decide whether retrieved chunks are relevant enough to
        answer from, based on the top relevance score vs. the configured
        threshold (`settings.RETRIEVAL_SCORE_THRESHOLD`).

        Args:
            chunks: Output of `retrieve`.

        Returns:
            True if the top chunk clears the relevance threshold.
        """
        if not chunks:
            return False
        return chunks[0].relevance_score >= settings.RETRIEVAL_SCORE_THRESHOLD
