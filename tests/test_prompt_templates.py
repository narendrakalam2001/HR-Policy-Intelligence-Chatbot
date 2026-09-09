"""tests/test_prompt_templates.py — prompt assembly and formatting."""

from __future__ import annotations

from app.prompt_templates import (
    OUT_OF_SCOPE_FALLBACK,
    SYSTEM_PROMPT,
    build_human_prompt,
    format_chat_history,
    format_context,
)


def test_format_context_empty_returns_placeholder() -> None:
    result = format_context([])
    assert "No relevant context" in result


def test_format_context_includes_source_and_page_citation() -> None:
    result = format_context([("Some policy text.", "leave_policy.pdf", 7)])
    assert "leave_policy.pdf" in result
    assert "p.7" in result
    assert "Some policy text." in result


def test_format_context_orders_multiple_excerpts() -> None:
    result = format_context(
        [
            ("First chunk.", "a.pdf", 1),
            ("Second chunk.", "b.pdf", 2),
        ]
    )
    assert result.index("First chunk.") < result.index("Second chunk.")


def test_format_chat_history_empty_returns_placeholder() -> None:
    assert "start of the conversation" in format_chat_history([])


def test_format_chat_history_renders_both_roles() -> None:
    history = [("What is gratuity?", "Gratuity is a lump sum benefit...")]
    result = format_chat_history(history)
    assert "User: What is gratuity?" in result
    assert "Assistant: Gratuity is a lump sum benefit..." in result


def test_build_human_prompt_includes_question_context_and_history() -> None:
    prompt = build_human_prompt(
        question="How many weeks of maternity leave?",
        context="[Excerpt 1] ...",
        chat_history="User: hi\nAssistant: hello",
    )
    assert "How many weeks of maternity leave?" in prompt
    assert "[Excerpt 1]" in prompt
    assert "User: hi" in prompt


def test_system_prompt_requires_citations_and_grounding() -> None:
    assert "Sources:" in SYSTEM_PROMPT
    assert "CONTEXT" in SYSTEM_PROMPT


def test_out_of_scope_fallback_is_non_empty() -> None:
    assert len(OUT_OF_SCOPE_FALLBACK) > 0
