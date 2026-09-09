"""tests/test_rag_pipeline.py — PDF discovery, extraction, and chunking."""

from __future__ import annotations

from pathlib import Path

from app.rag_pipeline import (
    build_documents,
    discover_pdfs,
    extract_pages_from_pdf,
    ingest_all,
)


def test_discover_pdfs_finds_the_sample(temp_policies_dir: Path) -> None:
    pdfs = discover_pdfs(str(temp_policies_dir))
    assert len(pdfs) == 1
    assert pdfs[0].name == "sample_gratuity_act.pdf"


def test_discover_pdfs_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert discover_pdfs(str(empty_dir)) == []


def test_discover_pdfs_missing_directory_returns_empty_list(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert discover_pdfs(str(missing)) == []


def test_extract_pages_from_pdf_returns_expected_text(temp_policies_dir: Path) -> None:
    pdf_path = discover_pdfs(str(temp_policies_dir))[0]
    pages = extract_pages_from_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].source_file == "sample_gratuity_act.pdf"
    assert "gratuity" in pages[0].text.lower()


def test_build_documents_attaches_source_and_page_metadata(temp_policies_dir: Path) -> None:
    pdf_path = discover_pdfs(str(temp_policies_dir))[0]
    pages = extract_pages_from_pdf(pdf_path)
    docs = build_documents(pages)

    assert len(docs) >= 1
    for doc in docs:
        assert doc.metadata["source_file"] == "sample_gratuity_act.pdf"
        assert doc.metadata["page_number"] == 1
        assert "chunk_id" in doc.metadata
        assert len(doc.metadata["chunk_id"]) == 24


def test_chunk_ids_are_deterministic_across_runs(temp_policies_dir: Path) -> None:
    pdf_path = discover_pdfs(str(temp_policies_dir))[0]
    pages = extract_pages_from_pdf(pdf_path)

    ids_run_1 = [d.metadata["chunk_id"] for d in build_documents(pages)]
    ids_run_2 = [d.metadata["chunk_id"] for d in build_documents(pages)]

    assert ids_run_1 == ids_run_2


def test_ingest_all_raises_on_empty_knowledge_base(tmp_path: Path, fake_embedding_function) -> None:
    from app.rag_pipeline import EmptyKnowledgeBaseError

    empty_dir = tmp_path / "empty_policies"
    empty_dir.mkdir()

    import pytest

    with pytest.raises(EmptyKnowledgeBaseError):
        ingest_all(
            policies_dir=str(empty_dir),
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_empty_kb",
        )


def test_ingest_all_actually_writes_to_disk(
    temp_policies_dir: Path, tmp_path: Path, fake_embedding_function
) -> None:
    """
    Regression test for a real bug: passing chromadb `client_settings`
    without `is_persistent=True` silently produces an in-memory-only
    client — ingestion reports success and the count is correct *within
    that process*, but nothing is ever written to disk, so a second
    process (e.g. `streamlit run` started after `python ingest.py`)
    finds an empty store. This checks the on-disk file directly, which
    the in-process `_collection.count()` check alone would not catch.
    """
    chroma_dir = tmp_path / "chroma"

    ingest_all(
        policies_dir=str(temp_policies_dir),
        persist_directory=str(chroma_dir),
        collection_name="test_disk_write",
    )

    sqlite_file = chroma_dir / "chroma.sqlite3"
    assert sqlite_file.exists(), (
        "chroma.sqlite3 was not created on disk — the Chroma client is "
        "running in-memory-only instead of persisting (check is_persistent "
        "is set on the client_settings passed to Chroma(...))"
    )
    assert sqlite_file.stat().st_size > 0


def test_ingest_all_persists_and_is_idempotent(
    temp_policies_dir: Path, tmp_path: Path, fake_embedding_function
) -> None:
    chroma_dir = str(tmp_path / "chroma")

    vector_store_1 = ingest_all(
        policies_dir=str(temp_policies_dir),
        persist_directory=chroma_dir,
        collection_name="test_idempotent",
    )
    count_after_first_run = vector_store_1._collection.count()
    assert count_after_first_run >= 1

    # Re-running ingestion on the same source PDFs must not duplicate chunks.
    vector_store_2 = ingest_all(
        policies_dir=str(temp_policies_dir),
        persist_directory=chroma_dir,
        collection_name="test_idempotent",
    )
    assert vector_store_2._collection.count() == count_after_first_run