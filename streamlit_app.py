"""
streamlit_app.py

Main chat interface for the HR Policy Intelligence Chatbot.

Run with:
    streamlit run streamlit_app.py

Assumes `python ingest.py` has already been run to populate the local
ChromaDB store from `data/policies/`.
"""

from __future__ import annotations

import logging

import streamlit as st

from app.llm_chain import ChatResponse, LLMCallError, RAGChatbot
from config import settings

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="HR Policy Intelligence Chatbot",
    page_icon="📄",
    layout="centered",
)


# --------------------------------------------------------------------------- #
# Cached resources — built once per server process, not per rerun
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading knowledge base and model...")
def load_chatbot() -> RAGChatbot | None:
    """
    Build the RAGChatbot once and cache it across Streamlit reruns.

    Returns:
        A ready RAGChatbot, or None if initialization failed (e.g. missing
        API key or empty vector store) — the caller renders the error.
    """
    try:
        settings.validate()
        return RAGChatbot()
    except Exception:
        logger.exception("Failed to initialize RAGChatbot.")
        return None


def init_session_state() -> None:
    """Initialize Streamlit session state on first load."""
    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str,
        #              "sources": list[tuple[str,int]], "confidence": float}
        st.session_state.messages = []
    if "chat_history_pairs" not in st.session_state:
        # Flat (user, assistant) pairs used as LLM conversation memory —
        # kept separate from `messages` (which also carries UI metadata
        # like sources/confidence that the LLM doesn't need to see).
        st.session_state.chat_history_pairs = []


def render_sources(sources: list[tuple[str, int]]) -> None:
    """Render a compact source-citation line under an assistant message."""
    if not sources:
        return
    formatted = "; ".join(f"{name}, p.{page}" for name, page in sources)
    st.caption(f"📎 Sources: {formatted}")


def render_confidence(confidence: float, is_fallback: bool) -> None:
    """Render a small confidence indicator under an assistant message."""
    if is_fallback:
        st.caption("🔸 Confidence: N/A (no relevant policy content found)")
        return
    if confidence >= 0.75:
        label, emoji = "High", "🟢"
    elif confidence >= settings.RETRIEVAL_SCORE_THRESHOLD:
        label, emoji = "Moderate", "🟡"
    else:
        label, emoji = "Low", "🔴"
    st.caption(f"{emoji} Confidence: {label} ({confidence:.2f})")


def render_history() -> None:
    """Re-render all prior messages on every Streamlit rerun."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []))
                render_confidence(msg.get("confidence", 0.0), msg.get("is_fallback", False))


def handle_user_input(chatbot: RAGChatbot, user_input: str) -> None:
    """Process one turn: append the user message, call the chatbot, append the reply."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            try:
                response: ChatResponse = chatbot.ask(
                    user_input, st.session_state.chat_history_pairs
                )
            except LLMCallError as exc:
                st.error(str(exc))
                logger.error("LLM call error surfaced to user: %s", exc)
                return

        st.markdown(response.answer)
        render_sources(response.sources)
        render_confidence(response.confidence, response.is_fallback)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "is_fallback": response.is_fallback,
        }
    )
    st.session_state.chat_history_pairs.append((user_input, response.answer))
    # Keep only the configured memory window client-side too, so the list
    # never grows unbounded over a very long session.
    st.session_state.chat_history_pairs = st.session_state.chat_history_pairs[
        -settings.CONVERSATION_MEMORY_TURNS :
    ]


def main() -> None:
    """Streamlit app entry point."""
    init_session_state()

    st.title("📄 HR Policy Intelligence Chatbot")
    st.caption(
        "Ask about leave policy, gratuity, maternity benefit, POSH, and other "
        "HR/labour-law topics covered in the ingested policy documents."
    )

    with st.sidebar:
        st.subheader("About")
        st.markdown(
            "This assistant answers **only** from the HR policy and labour "
            "law PDFs ingested into its knowledge base. It cites the source "
            "document and page for every answer, and will tell you when a "
            "question falls outside its knowledge base."
        )
        st.markdown(f"**LLM:** `{settings.GEMINI_MODEL}`")
        st.markdown(f"**Embedding model:** `{settings.EMBEDDING_MODEL_NAME}`")
        st.markdown(f"**Memory window:** last {settings.CONVERSATION_MEMORY_TURNS} turns")
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.chat_history_pairs = []
            st.rerun()

    chatbot = load_chatbot()
    if chatbot is None:
        st.error(
            "Chatbot failed to initialize. Check that:\n"
            "1. `GEMINI_API_KEY` is set in your `.env` file.\n"
            "2. You have run `python ingest.py` to build the knowledge base.\n\n"
            "See the terminal / logs for the full error."
        )
        st.stop()

    render_history()

    user_input = st.chat_input("Ask a question about HR policy...")
    if user_input:
        handle_user_input(chatbot, user_input)


if __name__ == "__main__":
    main()
