"""
tests/conftest.py

Shared pytest fixtures.

Tests use `FakeEmbeddings` (deterministic, random-vector embeddings from
langchain_community) instead of the real sentence-transformers model.
This keeps the suite fast, hermetic, and free of a network/model-download
dependency — the code path exercised (chunking, Chroma persistence,
metadata plumbing, retrieval API) is identical either way; only the
vector *content* differs, which these tests don't assert on.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

SAMPLE_TEXT = """\
GRATUITY ACT SUMMARY

Section 4: Gratuity is payable to an employee on the termination of his
employment after he has rendered continuous service for not less than
five years. The amount is calculated at fifteen days' wages for every
completed year of service.

Section 7: The employer shall pay the gratuity within thirty days from
the date it becomes payable.
"""


@pytest.fixture
def temp_policies_dir(tmp_path: Path) -> Path:
    """A temp directory containing one small, real, parseable PDF."""
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), SAMPLE_TEXT, fontsize=10)
    doc.save(str(policies_dir / "sample_gratuity_act.pdf"))
    doc.close()

    return policies_dir


@pytest.fixture
def fake_embedding_function(monkeypatch):
    """
    Patch `app.rag_pipeline.get_embedding_function` to return
    deterministic FakeEmbeddings instead of loading the real
    sentence-transformers model.
    """
    from langchain_community.embeddings import FakeEmbeddings

    from app import rag_pipeline

    fake = FakeEmbeddings(size=384)
    monkeypatch.setattr(rag_pipeline, "get_embedding_function", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _isolate_env():
    """
    Ensure tests never require a real Gemini key.

    `config.settings` is a frozen dataclass singleton read from the
    environment at import time (before pytest ever runs), so patching
    the env var alone has no effect on an already-constructed instance.
    `object.__setattr__` bypasses the frozen guard directly on the
    singleton, and the original value is restored after each test.
    """
    from config import settings

    original = settings.GEMINI_API_KEY
    object.__setattr__(settings, "GEMINI_API_KEY", original or "test-key-not-real")
    yield
    object.__setattr__(settings, "GEMINI_API_KEY", original)
