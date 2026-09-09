"""tests/test_vector_store.py — query preprocessing, retrieval, guardrails."""

from __future__ import annotations

import pytest

from app.rag_pipeline import ingest_all
from app.vector_store import Retriever, RetrievedChunk, preprocess_query


@pytest.mark.parametrize(
    "raw_query, expected_substring",
    [
        ("what is posh policy", "posh (prevention of sexual harassment)"),
        ("how many weeks of maternty leave", "maternity leave"),
        ("CL vs PL difference", "casual leave"),
        ("what about grattuity", "gratuity"),
    ],
)
def test_preprocess_query_expands_and_corrects(raw_query: str, expected_substring: str) -> None:
    result = preprocess_query(raw_query).lower()
    assert expected_substring in result


def test_preprocess_query_handles_blank_input() -> None:
    assert preprocess_query("   ") == ""


def test_preprocess_query_leaves_domain_terms_untouched() -> None:
    # "gratuity" is already correct and in-vocabulary; must not be altered.
    assert preprocess_query("gratuity eligibility") == "gratuity eligibility"


def test_is_in_scope_false_for_no_chunks() -> None:
    retriever = Retriever.__new__(Retriever)  # bypass __init__, no store needed for this check
    assert retriever.is_in_scope([]) is False


def test_is_in_scope_respects_threshold() -> None:
    retriever = Retriever.__new__(Retriever)

    high_relevance = [
        RetrievedChunk(text="x", source_file="f.pdf", page_number=1, relevance_score=0.9)
    ]
    low_relevance = [
        RetrievedChunk(text="x", source_file="f.pdf", page_number=1, relevance_score=0.01)
    ]

    assert retriever.is_in_scope(high_relevance) is True
    assert retriever.is_in_scope(low_relevance) is False


def test_retriever_returns_relevant_chunk_with_correct_metadata(
    temp_policies_dir, tmp_path, fake_embedding_function
) -> None:
    vector_store = ingest_all(
        policies_dir=str(temp_policies_dir),
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_retriever",
    )
    retriever = Retriever(vector_store=vector_store)

    chunks = retriever.retrieve("gratuity eligibility service years")

    assert len(chunks) >= 1
    assert chunks[0].source_file == "sample_gratuity_act.pdf"
    assert chunks[0].page_number == 1
    assert 0.0 <= chunks[0].relevance_score <= 1.0


def test_retriever_handles_empty_store_gracefully(tmp_path, fake_embedding_function) -> None:
    from langchain_chroma import Chroma

    from app.rag_pipeline import _build_chroma_client_settings

    empty_store = Chroma(
        collection_name="test_empty_retrieve",
        embedding_function=fake_embedding_function,
        persist_directory=str(tmp_path / "chroma_empty"),
        client_settings=_build_chroma_client_settings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    retriever = Retriever(vector_store=empty_store)

    chunks = retriever.retrieve("anything")

    assert chunks == []
    assert retriever.is_in_scope(chunks) is False