"""
app/llm_chain.py

Orchestration layer: ties together query preprocessing, retrieval,
guardrails, prompt assembly, and the Gemini LLM call into a single
`RAGChatbot` class used by streamlit_app.py.

Design notes:
    - Conversation memory is a simple bounded list of (user, assistant)
      tuples, capped at `settings.CONVERSATION_MEMORY_TURNS`. This is
      intentionally explicit rather than a LangChain memory object, so
      it's trivial to serialize into Streamlit's `st.session_state` and
      to reason about in an interview walkthrough.
    - The out-of-scope guardrail runs BEFORE any LLM call: if retrieval
      relevance is below threshold, we return the fallback message
      directly. This avoids paying for/waiting on a Gemini call we
      already know shouldn't be grounded, and prevents the model from
      being tempted to answer from general knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.prompt_templates import (
    OUT_OF_SCOPE_FALLBACK,
    SYSTEM_PROMPT,
    build_human_prompt,
    format_chat_history,
    format_context,
)
from app.vector_store import Retriever, RetrievedChunk, preprocess_query
from config import settings

logger = logging.getLogger(__name__)


class LLMCallError(RuntimeError):
    """Raised when the Gemini API call fails after retries are exhausted."""


@dataclass
class ChatResponse:
    """Structured result returned by `RAGChatbot.ask` for the UI to render."""

    answer: str
    sources: List[Tuple[str, int]]  # (source_file, page_number), deduped
    confidence: float  # 0.0 - 1.0, based on top retrieval relevance
    is_fallback: bool  # True if answered via the out-of-scope guardrail
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)


def _build_llm() -> ChatGoogleGenerativeAI:
    """
    Construct the Gemini chat model client.

    Reads all tunables from `config.settings` — nothing hardcoded here —
    so swapping model name/temperature never requires touching this file.
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "Google Gemini API key before starting the chatbot."
        )
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=settings.GEMINI_TEMPERATURE,
        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
        # Bound how long a single call can hang on a network issue. The
        # underlying gRPC transport otherwise retries transport-level
        # connection failures internally without raising, which can hang
        # a Streamlit request indefinitely on a flaky network.
        timeout=settings.GEMINI_REQUEST_TIMEOUT_SECONDS,
    )


class RAGChatbot:
    """
    Stateless-per-call RAG chatbot: conversation history is passed in and
    returned by the caller (Streamlit session state) rather than held as
    mutable internal state, so a single instance is safe to reuse across
    Streamlit reruns and sessions.
    """

    def __init__(self, retriever: Retriever | None = None) -> None:
        """
        Args:
            retriever: Optional pre-built Retriever (for tests). Defaults
                to a Retriever backed by the persisted production store.
        """
        self._retriever = retriever or Retriever()
        self._llm = _build_llm()

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _call_llm(self, system_prompt: str, human_prompt: str) -> str:
        """
        Call Gemini with retry/backoff (transient network or rate-limit
        errors are common on the free tier and should not surface as a
        hard failure to the end user on the first blip).
        """
        try:
            response = self._llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            return response.content if isinstance(response.content, str) else str(response.content)
        except Exception:
            logger.exception("Gemini API call failed.")
            raise

    def ask(
        self,
        raw_question: str,
        chat_history: List[Tuple[str, str]],
    ) -> ChatResponse:
        """
        Answer a user question grounded in the HR policy knowledge base.

        Args:
            raw_question: Exact text the user typed.
            chat_history: Full conversation so far as (user, assistant)
                tuples, oldest first. This method truncates it to the
                configured memory window internally.

        Returns:
            A ChatResponse with the answer, cited sources, a confidence
            score, and whether the guardrail fallback was used.
        """
        question = preprocess_query(raw_question)

        chunks = self._retriever.retrieve(question)
        top_score = chunks[0].relevance_score if chunks else 0.0

        if not self._retriever.is_in_scope(chunks):
            logger.info(
                "Query treated as out-of-scope (top score %.3f < threshold %.3f): %r",
                top_score,
                settings.RETRIEVAL_SCORE_THRESHOLD,
                raw_question,
            )
            return ChatResponse(
                answer=OUT_OF_SCOPE_FALLBACK,
                sources=[],
                confidence=top_score,
                is_fallback=True,
                retrieved_chunks=chunks,
            )

        windowed_history = chat_history[-settings.CONVERSATION_MEMORY_TURNS :]
        context = format_context(
            [(c.text, c.source_file, c.page_number) for c in chunks]
        )
        history_text = format_chat_history(windowed_history)
        human_prompt = build_human_prompt(question, context, history_text)

        try:
            answer_text = self._call_llm(SYSTEM_PROMPT, human_prompt)
        except Exception as exc:
            raise LLMCallError(
                "The assistant is temporarily unavailable (Gemini API error). "
                "Please try again in a moment."
            ) from exc

        sources = sorted(
            {(c.source_file, c.page_number) for c in chunks},
            key=lambda pair: (pair[0], pair[1]),
        )

        return ChatResponse(
            answer=answer_text,
            sources=sources,
            confidence=top_score,
            is_fallback=False,
            retrieved_chunks=chunks,
        )
