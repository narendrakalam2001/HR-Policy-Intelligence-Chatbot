"""
tests/test_llm_chain.py — RAGChatbot orchestration.

The Gemini API itself is never called in these tests: the out-of-scope
guardrail test verifies the LLM is never reached, and the in-scope test
stubs `RAGChatbot._call_llm` directly. This keeps the suite hermetic
(no API key, no network, no cost) while still exercising every branch
of `RAGChatbot.ask`.
"""

from __future__ import annotations

from typing import List, Tuple

from config import settings

from app.llm_chain import RAGChatbot, _extract_answer_text
from app.vector_store import Retriever, RetrievedChunk


class _StubRetriever(Retriever):
    """A Retriever double that returns a fixed set of chunks, no real store."""

    def __init__(self, chunks: List[RetrievedChunk], in_scope: bool) -> None:  # noqa: super-init-not-called
        self._chunks = chunks
        self._in_scope = in_scope

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        return self._chunks

    def is_in_scope(self, chunks: List[RetrievedChunk]) -> bool:
        return self._in_scope


def _make_bot(chunks: List[RetrievedChunk], in_scope: bool) -> RAGChatbot:
    retriever = _StubRetriever(chunks, in_scope)
    return RAGChatbot(retriever=retriever)


def test_out_of_scope_query_never_calls_the_llm(monkeypatch) -> None:
    bot = _make_bot(chunks=[], in_scope=False)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be called for an out-of-scope query")

    monkeypatch.setattr(bot, "_call_llm", _fail_if_called)

    response = bot.ask("What's the weather today?", chat_history=[])

    assert response.is_fallback is True
    assert response.sources == []
    assert response.confidence == 0.0


def test_in_scope_query_calls_llm_and_returns_sources(monkeypatch) -> None:
    chunks = [
        RetrievedChunk(
            text="Gratuity is payable after five years of service.",
            source_file="gratuity_act.pdf",
            page_number=2,
            relevance_score=0.85,
        ),
        RetrievedChunk(
            text="The amount is 15 days' wages per year of service.",
            source_file="gratuity_act.pdf",
            page_number=3,
            relevance_score=0.80,
        ),
    ]
    bot = _make_bot(chunks=chunks, in_scope=True)

    captured_prompts: dict = {}

    def _fake_call_llm(system_prompt: str, human_prompt: str) -> str:
        captured_prompts["system"] = system_prompt
        captured_prompts["human"] = human_prompt
        return "Gratuity is payable after 5 years of service. Sources: gratuity_act.pdf, p.2"

    monkeypatch.setattr(bot, "_call_llm", _fake_call_llm)

    response = bot.ask("How many years for gratuity?", chat_history=[])

    assert response.is_fallback is False
    assert response.confidence == 0.85
    assert ("gratuity_act.pdf", 2) in response.sources
    assert ("gratuity_act.pdf", 3) in response.sources
    assert "gratuity" in captured_prompts["human"].lower()


def test_conversation_history_is_windowed_to_configured_turns(monkeypatch) -> None:

    chunks = [
        RetrievedChunk(text="x", source_file="f.pdf", page_number=1, relevance_score=0.9)
    ]
    bot = _make_bot(chunks=chunks, in_scope=True)

    captured: dict = {}

    def _fake_call_llm(system_prompt: str, human_prompt: str) -> str:
        captured["human"] = human_prompt
        return "answer"

    monkeypatch.setattr(bot, "_call_llm", _fake_call_llm)

    # Build more turns than the configured memory window.
    long_history: List[Tuple[str, str]] = [
        (f"question {i}", f"answer {i}") for i in range(settings.CONVERSATION_MEMORY_TURNS + 5)
    ]
    bot.ask("latest question", chat_history=long_history)

    # Only the most recent N turns should appear in the prompt.
    oldest_turn = long_history[0][0]
    assert oldest_turn not in captured["human"]
    newest_kept_turn = long_history[-1][0]
    assert newest_kept_turn in captured["human"]


# --------------------------------------------------------------------------- #
# Regression tests for _extract_answer_text
# --------------------------------------------------------------------------- #
#
# Real bug: Gemini 3+ "thinking" models sometimes return `.content` as a
# list of blocks (one for internal reasoning, one for the final answer)
# instead of a plain string. Naively str()-ing that list leaked the raw
# Python repr of the model's internal reasoning into the chat UI — e.g.
# "['Professional/concise? Yes...', 'Based on the provided documents...']".
# This happens intermittently (thinking isn't triggered on every call), so
# these tests pin both shapes down explicitly rather than relying on ever
# reproducing it live.


def test_extract_answer_text_handles_plain_string() -> None:
    assert _extract_answer_text("Gratuity is payable after 5 years.") == (
        "Gratuity is payable after 5 years."
    )


def test_extract_answer_text_filters_out_thinking_blocks() -> None:
    content = [
        {"type": "thinking", "thinking": "Let me check the tone and disclaimer rules..."},
        {"type": "text", "text": "Gratuity is payable after 5 years.", "extras": {"signature": "x"}},
    ]
    assert _extract_answer_text(content) == "Gratuity is payable after 5 years."


def test_extract_answer_text_handles_thought_key_variant() -> None:
    content = [
        {"thought": True, "text": "internal reasoning, not the answer"},
        {"text": "The final answer."},
    ]
    assert _extract_answer_text(content) == "The final answer."


def test_extract_answer_text_joins_multiple_text_blocks() -> None:
    content = [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]
    assert _extract_answer_text(content) == "Part one. Part two."


def test_extract_answer_text_never_leaks_raw_list_repr() -> None:
    """The exact failure mode this bug caused: a list ending up as a raw
    Python-repr string in the answer, e.g. starting with "[' " or "[{'"."""
    content = [
        {"type": "thinking", "thinking": "Professional/concise? Yes."},
        {"type": "text", "text": "Under the Maternity Benefit Act..."},
    ]
    result = _extract_answer_text(content)
    assert not result.startswith("[")
    assert "Professional/concise" not in result